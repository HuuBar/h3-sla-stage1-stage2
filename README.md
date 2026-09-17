# MiniMax-H3 SLA 稀疏注意力微调代码包（stage1 + stage2，完整版）

SLA(Sparse-Linear Attention, arXiv:2509.24006)接到 DiT 上做少量全参微调，让模型适应块级稀疏注意力
(~95% 稀疏)。本包是 DiffSynth(昇腾 NPU)上 **MiniMax-H3 T2VA 音视频 DiT** 两阶段微调的全部代码、
启动脚本、配置与踩坑记录，可整体搬到别的仓库。

- 上游基座：`minimax_dmd`(DiffSynth 派生) @ `91b83d8`(2026-08-10)
- 新增 4 个文件；改动 14 个文件（`patches/sla-finetune.patch`，421 行）
- 最后一次实跑：stage1 单卡 1000 步 → 合并成 `newcont309-proj1000` → stage2 全量 4959 条 / ~310 步
  （topk 0.05，768×1344×124，ZeRO-3 16 卡 + CPU offload）

## 两阶段是什么

| 阶段 | 做什么 | 产物 |
|---|---|---|
| **stage1** | 冻结主干，**只训 `proj_l`**(50×128×128 ≈ 0.83M)；loss = 每层 `o_sla` vs 全注意力 `o_full` 的 MSE（teacher 对齐）|
| **merge** | 把 proj_l 覆盖写回主干 | 635 keys 完整 ckpt（66GB） |
| **stage2** | **联合微调**：主干 lr 1e-5 / proj_l 5e-5，标准 flow-matching SFT | 完整 ckpt |

---

## 目录结构

```
files/                                     ← 按仓库相对路径排放，直接 `cp -r files/* <你的仓库>/`
  diffsynth/models/sla_core.py             SLA 模块（稀疏分支 + 线性分支 proj_l）
  diffsynth/models/sla_kernel.py           块稀疏 kernel（forward + torch 重算反向）
  diffsynth/models/sla_utils.py            块 mean-pool 打分 + top-k 选块
  diffsynth/models/sla_reference.py        纯 torch 参考实现（数值对拍）
  diffsynth/models/minimax_h3_dit.py       SLA 挂载点（换 DiT 只改这里）
  diffsynth/models/model_loader.py         【改】┐
  diffsynth/core/loader/config.py          【改】├ extra_kwargs 透传
  diffsynth/diffusion/base_pipeline.py     【改】┘
  diffsynth/core/loader/model.py           【改】缺 proj_l 时放宽 load_state_dict strict
  diffsynth/core/offload_training/manager.py 【改】torch.cuda.synchronize → NPU 分支
  diffsynth/core/data/operators.py         【改】音频加载 torchaudio → soundfile 回退
  diffsynth/configs/model_configs.py       【改】注册 SLA ckpt 的 model_hash
  diffsynth/diffusion/loss.py              【改】TeacherAlign 逐层对齐 loss + sigma 加权
  diffsynth/diffusion/training_module.py   【改】task 判定 startswith("sft:train")，覆盖 train_align
  diffsynth/diffusion/runner.py            【改】双 lr + 每步 empty_cache + sla_sparsity 日志
  diffsynth/diffusion/flow_match.py        【改】┐ sigma_weight_cap 属性
  diffsynth/diffusion/ddim_scheduler.py    【改】┘（loss.py 会读，缺了报 AttributeError）
  examples/.../model_training/train.py     【改】训练入口（全部 SLA 超参 + 阶段开关）
  examples/.../full/accelerate_config_zero3_16gpu_offload.yaml    stage2 / stage1-16卡
  examples/.../full/accelerate_config_single_gpu.yaml             stage1 单卡
patches/sla-finetune.patch                 14 个改动文件的 git diff（与 files/ 二选一）
scripts/
  stage1/  train_newcont_proj_1g_128.sh        ← stage1 单卡（--proj-only --teacher-align）
           train_newcont_proj_16g_128.sh       ← stage1 16 卡版
           launch_proj_1g_130.sh               ← stage1 nohup 启动器
           train_proj_only.py                  ← stage1 独立实现（两遍法，不依赖 accelerate/zero3）
           run_proj_train.sh                   ← 上面那个的启动器
           merge_proj_into_backbone.py         ← proj_l 合并回主干（stage2 起点）
  stage2/  train_backbone_full_130.sh          ← 最后一次全量微调
           train_backbone_full_run_130.sh      ← 其 nohup 启动器
           prep_stage2_full_128.py             ← 构建 stage2 数据缓存（4959 条）
  reference_108/  更早几轮的 768p 启动脚本（opencomp 156 / proj-156 / proj-1000 / merge_sla_dmd 等）
  container/      训练容器启动脚本（设备 + Ascend 驱动挂载）
docs/
  sla-two-stage-projonly-20260817.md    两阶段设计/踩坑全记录
  sla-stage1-singlegpu-20260817.md      stage1 为什么走单卡
  sla_h3_sparse_training_report.md      稀疏方案技术报告（列保护 vs 开放竞争实测）
  h3_sla_training_log.md                到 8/15 的训练全记录 + 白屏根因链
  sla_retrain_plan.md / h3_sla_training_plan.md   重训方案 / 原始计划
```

## stage1 怎么跑

```bash
cd <repo>
# A) 走训练框架（单卡）
SLA_FORCE_OFF=1 ASCEND_RT_VISIBLE_DEVICES=0 bash scripts/stage1/train_newcont_proj_1g_128.sh
# B) 独立两遍法脚本（不依赖 accelerate/zero3，实测 ~19 s/it，200 步约 64 分钟）
python3 scripts/stage1/train_proj_only.py --steps 200 --lr 1e-4 --out <输出目录>
```
核心参数：
```
--proj-only --teacher-align --align-timesteps 50 --learning_rate 1e-4
--task "sft:train_align" --use_sla --sla_topk 0.05 --sla_feature_map softmax --sla_blkq 64 --sla_blkk 64
```
环境变量：`SLA_PROJ_ONLY=1`（主干输出 detach、不跑稀疏反向）、`SLA_ALIGN_TEACHER=1`（每层累加对齐 MSE 并逐层 backward）。
输出只有 proj_l（`step-N.safetensors` / `proj_l_stepN.pt`，100 keys）。

**合并成 stage2 起点：**
```bash
python3 scripts/stage1/merge_proj_into_backbone.py <主干.safetensors> <proj_l.safetensors> <输出.safetensors>
```

## stage2 怎么跑

```bash
cd <repo>
SLA_FORCE_OFF=1 bash scripts/stage2/train_backbone_full_130.sh
# 后台：bash scripts/stage2/train_backbone_full_run_130.sh   → dataset/train_backbone_full.log
```
核心参数：
```
--use_sla --sla_topk 0.05 --sla_feature_map softmax --sla_blkq 64 --sla_blkk 64
--learning_rate 1e-5 --proj-lr 5e-5 --task "sft:train" --save_steps 310
```
`SLA_FORCE_OFF=1` = 开放竞争（去掉文本/音频 key 块的列保护，纯全局 topk）；不设则走保护版（稀疏度 0.9513 → 0.9294）。

## 数据格式

`--dataset_base_path` 下每个 `.pth` 是 `(shared, posi, nega)`，训练只用前两个：
```python
shared["input_latents"]        # (1, 24, 37, 48, 84) bf16  768p 视频 VAE latent
shared["audio_input_latents"]  # (2, 32, 207) bf16         音频 latent
shared["use_gradient_checkpointing"]   # 缓存里的值优先于 CLI，要改建缓存
posi["prompt_embeds"]          # (343, 5120) bf16          文本条件
posi["packed"]                 # img_pos/audio_pos/text_pos/img_position_ids/token_tags/cu_seqlens/seq_len(=38080)
```

## 注意

1. **`use_sla` 命中检查要求模型路径含 `transformer`**（`train.py:52`）：起点 ckpt 先软链成
   `dataset/transformer_xxx_link.safetensors`，否则按稠密构造，SLA 静默失效。
2. **ZeRO-3 下不能用 `--resume_from_checkpoint`**（每 rank 只有分片，size mismatch）；起点走 `--model_paths`。
3. **teacher 对齐必须关 gradient checkpointing**：重算 forward 会二次触发每层 backward，梯度重复累加。
4. **stage1 16 卡有风险**：bf16 主干全冻结 + 只有 fp32 proj_l 有梯度，会触发 deepspeed 0.19.4 的梯度分桶(ds_id)断言；
   单卡版（proj_l 仅 3MB）是已验证路线。

许可：`sla_*.py` 来自 SLA 官方，Apache-2.0，引用 `arXiv:2509.24006`。
