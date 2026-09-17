# MiniMax-H3 SLA 稀疏注意力微调代码

SLA(Sparse-Linear Attention, arXiv:2509.24006)接到 DiT 上做少量全参微调，让模型适应块级稀疏注意力
(~95% 稀疏)。本包是 DiffSynth(昇腾 NPU)上 **MiniMax-H3 T2VA 音视频 DiT** 两阶段微调的全部代码、
启动脚本、配置与踩坑记录。
- 最后一次实跑：stage1 单卡 1000 步 → 合并成 `newcont309-proj1000` → stage2 全量 4959 条 / ~310 步
  （topk 0.05，768×1344×124，ZeRO-3 16 卡 + CPU offload）

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
  diffsynth/models/minimax_h3_dit.py       SLA 挂载点（换 DiT 只改这里）
  diffsynth/core/offload_training/manager.py  torch.cuda.synchronize → NPU 分支
  diffsynth/configs/model_configs.py       注册 SLA ckpt 的 model_hash + SLA 构造参数(extra_kwargs)
  diffsynth/diffusion/loss.py              TeacherAlign 逐层对齐 loss
  diffsynth/diffusion/training_module.py   task 判定 startswith("sft:train")，覆盖 train_align
  diffsynth/diffusion/runner.py            双 lr + 每步 empty_cache + sla_sparsity 日志
  examples/.../model_training/train.py     训练入口（全部 SLA 超参 + 阶段开关）
  examples/.../full/accelerate_config_zero3_16gpu_offload.yaml    stage2 / stage1-16卡
  examples/.../full/accelerate_config_single_gpu.yaml             stage1 单卡
scripts/
  stage1/  train_newcont_proj_1g_128.sh        ← stage1 单卡（--proj-only --teacher-align）
           train_newcont_proj_16g_128.sh       ← stage1 16 卡版（未跑通）
           launch_proj_1g_130.sh               ← stage1 nohup 启动器
           merge_proj_into_backbone.py         ← proj_l 合并回主干（stage2 起点）
  stage2/  train_backbone_full_130.sh          ← 最后一次全量微调
           train_backbone_full_run_130.sh      ← 启动器
           prep_stage2_full_128.py             ← 构建 stage2 数据缓存（4959 条）
  container/      训练容器启动脚本（设备 + Ascend 驱动挂载）
```

## stage1 怎么跑

```bash
cd <repo>
SLA_FORCE_OFF=1 ASCEND_RT_VISIBLE_DEVICES=0 bash scripts/stage1/train_newcont_proj_1g_128.sh
```
核心参数：
```
--proj-only --teacher-align --align-timesteps 50 --learning_rate 1e-4
--task "sft:train_align" --use_sla --sla_topk 0.05 --sla_feature_map softmax --sla_blkq 64 --sla_blkk 64
```
环境变量：`SLA_PROJ_ONLY=1`（主干输出 detach、不跑稀疏反向）、`SLA_ALIGN_TEACHER=1`（每层累加对齐 MSE 并逐层 backward）。
输出只有 proj_l（`step-N.safetensors`，100 keys）。

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
5. **SLA 构造参数写在 `configs/model_configs.py` 的 `extra_kwargs` 里**（上游 loader 走 `model_class(**extra_kwargs)`），
   不走命令行透传：`--use_sla / --sla_topk / --sla_feature_map / --sla_blkq / --sla_blkk` 仍被接受但**不生效**，
   改 topk 或块大小要改注册表那条 `extra_kwargs`；想完全关掉 SLA 就删掉它。
