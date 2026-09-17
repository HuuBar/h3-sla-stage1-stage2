# SLA stage1 (proj-only) 单卡训练实装 — 2026-08-17 晚

用户两阶段定案: stage1 冻结主干只训 proj_l → stage2 解冻联合 (主干 1e-5 / proj_l 5e-5)。
本文档 = stage1 单卡执行的完整配方 (验证到第一步, 含三个必改坑)。

## 为什么 stage1 只能单卡 + 独立脚本

1. **16 卡 zero3 崩**: "bf16 主干全冻结 + 仅 fp32 proj_l 有梯度" 触发 deepspeed zero3
   `AssertionError: len(set(p.ds_id for p in params_in_bucket)) == len(params_in_bucket)`
   (reduce_independent_p_g_buckets_and_remove_grads, stage3.py:1508)。stage2 全参训练
   (混合 dtype 都训) 无此问题。proj_l 是 fp32 (sla_core.py:151 `nn.Linear(head_dim,
   head_dim, dtype=torch.float32)`)。
2. **train.py 单卡也崩**: accelerate_config_single_gpu.yaml (distributed_type NO,
   num_processes 1) + `--enable_model_cpu_offload` → `NotImplementedError: Cannot copy
   out of meta tensor; no data!` (torch_npu/utils/_module.py:76 convert)。train.py 框架
   单卡非 deepspeed 时模型走 meta 占位初始化, 无人填数据。
3. **解法 = 独立脚本** `/mnt/a800_share/c50058431/tools/train_stage1_projonly_1gpu.py`:
   复用推理脚本已验证的加载路径 (MiniMaxH3Pipeline.from_pretrained + ModelConfig(**cpu_vram)
   + vram_limit=npu_total-8), 手动训练循环喂 TeacherAlignMiniMaxH3AudioVideoLoss
   (import from diffsynth.diffusion.loss), 梯度在 sla_core 层内逐层 backward 累加,
   只对 proj_params 做 optimizer.step()。

## 独立脚本三个必改点 (每个都实测崩过一次)

1. **proj_l VRAMLinear meta 占位**: 原始 FL2VA 无 proj_l 权重 → CPU offload 包装
   (VRAMLinear) 后 weight 是 meta, forward 到 `cast_to` 崩
   (`NotImplementedError: Cannot copy out of meta tensor`)。加载后显式 materialize:
   ```python
   for blk in pipe.dit.blocks:
       sla = getattr(blk.attn, "sla_module", None)
       if sla is not None and hasattr(sla, "proj_l"):
           pl = sla.proj_l
           w = getattr(pl, "weight", None)
           if w is not None and w.is_meta:
               pl.weight = torch.nn.Parameter(torch.zeros(w.shape, dtype=torch.float32))
           b = getattr(pl, "bias", None)
           if b is not None and b.is_meta:
               pl.bias = torch.nn.Parameter(torch.zeros(b.shape, dtype=torch.float32))
   ```
2. **packed 嵌套 dict 必须递归转 device**: 单层循环 `for k,v in inputs.items()` 漏掉
   embeds["packed"] 里的 img_pos → `index_copy_ error: index on cpu, tensor on npu:0`。
   用递归 _to_device (Tensor→.to(DEVICE)+dtype, dict/list/tuple 递归)。
3. **scheduler 50 点网格**: `set_timesteps(50, training=True, shift=12.0)` (audio 同,
   shift=3.0) — 与 train.py teacher_align 模式一致 (训练分布=推理分布), 不是 1000。

其他要点:
- 冻结: `param.requires_grad_(keep)` where keep = "sla_module.proj_l" in name;
  正确打印 = "100 个 proj_l 参数对象, 825600 params 可训练" (50 blocks × weight+bias,
  128×128×50×2 + bias 50×128... 以实际为准)。
- env: `SLA_PROJ_ONLY=1` + `SLA_ALIGN_TEACHER=1` (sla_core 读)。
- inputs 组装: `dict(shared)` + `embeds` 合并, 强制
  `use_gradient_checkpointing=False` (teacher align 必须关, checkpoint 重算会二次
  backward 梯度重复累加), `max/min_timestep_boundary = 1.0/0.0`。
- 数据: 200 条子集缓存 `ir5k_gen_latent_cache_480p_ckpt_200` (从 _2000 前 200 个软链接),
  单卡 repeat 1 = 200 步; lr 1e-4, AdamW(proj_params)。
- 保存: 只导出 proj_l (`torch.save({name: p.detach().cpu() ...}, OUT+".pt")`),
  主干零漂移 = 原始 FL2VA。

## 并行分工 (用户定)

.28 一台机器同时跑:
- npu0: 单卡流式推理 (infer_sla_v2_cafe.py, set_device(0))
- npu1: stage1 训练 (`export ASCEND_RT_VISIBLE_DEVICES=1` + 独立脚本)
.4 的 stage2 联合训练 (16 卡 offload) 不动。

## 监视

`/mnt/a800_share/c50058431/tools/stage1_monitor.sh` — ssh .28 + ps python3 + npu-smi
卡1, 每 15s 刷新。用户偏好单层命令, 嵌套引号 watch 跑不通。

## 时间线

- 16 卡第一次崩: TeacherAlign loss detach → deepspeed "loss must be a scalar tensor"
  (已修: align_mean + zero_grad_anchor, 见 SKILL.md 两阶段节)。
- 16 卡第二次崩: zero3 ds_id bucket 断言 → 转单卡。
- train.py 单卡: meta tensor → 转独立脚本。
- 独立脚本 v1: img_pos cpu/npu 错位 → 递归 _to_device。
- 独立脚本 v2: proj_l meta → materialize。
- v3: 第一步执行中 (50 点网格逐层对齐, 单步慢正常)。

## ⚠️⚠️ v3 路线实测不可行 (2026-08-18 实锤) — 正解是 8/15 两遍法

**SLA_ALIGN_TEACHER 路径 (TeacherAlignMiniMaxH3AudioVideoLoss, 每层 sla forward 内立即
backward) 在单卡 CPU offload 下第一步 35 分钟还没完成 — 200 步 = 4.4 天, 不可行。**
死亡特征: AICore 42% 不涨 (NPU 没吃饱)、主线程 wchan=hrtimer_nanosleep (sleep 等 NPU)、
`/proc/<pid>/io` syscr 117 万次 read 调用 = offload 状态机在疯狂搬运参数 (CPU↔NPU thrash,
backward 要参数留卡 vs forward 后 offload 的调度冲突)。推理同机制 8.4s/it 没问题, 是训练
backward 改变了 offload 行为。

**正解 = 8/15 的 train_proj_only.py 两遍法 (验证跑通 200 步, 64 分钟, ~19s/步)**:
forward 全程 `with torch.no_grad()` (显存=推理级, 10.9GB), 用
`register_forward_pre_hook` 抓每层 `o_full` (全注意力 teacher) + `register_forward_hook`
抓 `o_sla`/`o_l` (proj_l 前输出, sla_core SLA_PROJ_ONLY=1 时 `self._o_l` 保存) **全部搬到
CPU**, 然后**只对 proj_l 分支逐层重算**: `o_sla = o_s + proj(o_l)` → `mse(o_sla, o_full)`
→ `loss_i.backward()` (每层独立小图, 图用完即释放)。主干全程无梯度, 不触发 offload thrash。

- **脚本**: `/mnt/a800_share/c50058431/tools/train_proj_only_200.py` (从 8/15 的
  train_proj_only.py 复制, 只改 CACHE 到平铺缓存 + glob `*.pth`)。
  8/15 原版 CACHE 是 3600 split-cache 子目录结构 (glob `*/*.pth`), 新 200 条缓存是平铺
  (glob `*.pth`) — **两个格式的 glob 必须跟着缓存结构改**。
- **启动**: `python3 train_proj_only_200.py --steps 200 --lr 1e-4
  --out /mnt/a800_share/c50058431/proj_l_align_200 --save-every 50`
  (ASCEEND_RT_VISIBLE_DEVICES=1 单卡)。
- **结果 (实测)**: proj|w| 0.0001→0.0062 持续增长 (200 步未饱和), loss 震荡下降
  (27→13.8~24 区间, pairs=50 每层对齐), mean|w| 0.0049 — 与 8/15 的 proj_l_step250
  (0.0053) 同量级。产物 proj_l_step200.pt (100 keys = 50 blocks × weight/bias)。
- **转 stage2 注入**: `torch.save` 的 .pt 不能直接喂 model_paths, 用
  `from safetensors.torch import save_file; save_file(sd, "proj_l_step200.safetensors")`。
  stage2 脚本 MODEL_PATHS_DIT 恢复 `json.dumps([tr + [proj]])` 合并注入 +
  `--proj-lr 5e-5` (双 lr)。stage2 16 卡 zero3 实测 83.8s/it 稳定 (无 OOM, 无 bucket 断言
  — stage2 全参训练本来就没那问题)。
- **判别"在跑还是 thrash"**: 单步 >10 min + AICore 不涨 + syscr 百万级 = thrash, 直接止损
  换两遍法, 别等第一步。两遍法第一步 135s (含编译预热), 之后 ~19-20s/步。

## 8/18 全天最终状态

- stage1 (两遍法): ✅ 200 步完成, proj_l_step200.pt (mean|w|=0.0049)。
- stage2 (两阶段接力): .4 16 卡, proj_l 注入 step200 + 主干 1e-5 / proj_l 5e-5,
  125 步 ~83.8s/it 稳定训练中, 输出 MiniMax-H3-T2VA-SLA-stage2-step200-2stage。
- 用户记忆中的"不分 stage 的 .4 版本" = MiniMax-H3-T2VA-SLA-3600-proj (8/15-16,
  单阶段联合, proj_l 注入 step250 0.0053, 450 步完成, loss 0.17→0.99 震荡), 当时推过
  golden_retriever 对比组 (cmp3_ckpt-proj 等) + cafe_ckpt0.mp4。回答训练进度问题时
  三版都要能对上号: 3600-proj (旧, 450步) / stage2-gen2000-proj0-lr1e5 (8/17, 125步) /
  stage2-step200-2stage (8/18 两阶段, 125步)。
