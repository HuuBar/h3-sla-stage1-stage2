# SLA 两阶段训练定案 + v2 重跑 + 单卡推理 — 2026-08-17 晚

承接 sla-stage2-sftloss-gen2000-20260817.md。该文件记录的 v1(proj=0.0053 注入 +
lr 3e-6 + 普通 SFT loss)实测结果不佳, 用户当晚拍板两阶段方案。

## v1 实测结果 (为什么推翻)

- 训练: 2000 条生成 latent, proj_l 注入 step250 (mean|w|≈0.0053), lr 3e-6,
  普通 SFT loss, 16 卡 zero3 + offload + empty_cache (修复内存 OOM 后稳定跑到 50+ 步)。
- step-50 推理 (cafe prompt, 50 步, diffsynth 单卡): **用户目检"噪声很大"**。
- tensorboard 证据: loss 首 0.769 末 0.523, 但分窗剧烈震荡
  (step1=0.769, 11=0.006, 21=0.911, 31=0.153, 41=0.736, 51=0.097) —
  不收敛。对照 dense ir5k-gen50 (同数据同 50 步): 0.84→0.71 平滑下降。
  sla_sparsity 稳定 0.90 (topk=0.05 预期, 含 force 列保护) — 稀疏机制本身正常。
- 结论: 噪声大 = SLA loss 不稳定未收敛。疑似 proj_l 从 0.0053 起步 + lr 3e-6
  与主干耦合时数值不稳。**用户定案: proj 改 0 (零初始化), lr 改 1e-5 重跑。**

## v2 重跑 (proj=0 + lr=1e-5)

- proj_l 用 SLA 默认零初始化: sla_core.py:182-183 `nn.init.zeros_(self.proj_l.weight/bias)`
  — 即 **model_paths 不注入 proj_l_step250.safetensors**, 只传 transformer 13 shards。
- stage2.sh 改动: `LR=3e-6`→`1e-5`; MODEL_PATHS_DIT 生成去掉 proj 文件
  (`print(json.dumps([tr]))` 而非 `[tr + [proj]]`); OUT→
  `MiniMax-H3-T2VA-SLA-stage2-gen2000-proj0-lr1e5`。
- ⚠️ 注意: 之前注入 step250 的合并 hash 922fd0b8 只在 14 文件配置时匹配;
  纯 13 shards 配置 hash 不同但 zero3 下仍可加载 (实测过)。

## 两阶段方案 (用户定案, "最稳的改法")

阶段一: 冻结主干只训 proj_l (lr 1e-4, ~200 步), 让线性分支先补偿稀疏化丢失的
长尾信息, 主干零漂移。阶段二: 解冻主干联合训 (主干 lr 1e-5, proj_l 5e-5),
起点是补偿到位的 proj_l。

### train.py 新增参数 (2026-08-17 晚实现)

- `--proj-only`: 阶段一用。__init__ 里遍历 `self.pipe.dit.named_parameters()`,
  只有名字含 `sla_module.proj_l` 的置 `requires_grad=True`, 其余冻结;
  同时 `os.environ["SLA_PROJ_ONLY"]="1"` — sla_core.py:220-228 读到后
  `o_l = o_l.detach()` 保存原始线性输出 + `o_s = o_s.detach()` (主干零漂移,
  sparse backward 不再执行, 省显存省时间, 数学等价)。
- `--proj-lr`: 阶段二双 lr 用。`self.proj_lr = proj_lr` 挂在 training module 上,
  runner.py 读它构造参数组。
- 传递链路: parser.add_argument → MiniMaxH3TrainingModule(...) 传参 →
  __init__ 存 self.proj_lr。task_to_loss 里 "sft:train_align" 只有
  `--teacher-align` 时注册 (TeacherAlignMiniMaxH3AudioVideoLoss, loss.py:103)。

### runner.py 双 lr (optimizer 参数组)

```python
proj_lr = getattr(model, "proj_lr", None)
if proj_lr is not None:
    main_params, proj_params = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (proj_params if "sla_module.proj_l" in name else main_params).append(p)
    optimizer = optimizer_class([
        {"params": main_params, "lr": learning_rate},
        {"params": proj_params, "lr": proj_lr},
    ], weight_decay=weight_decay)
else:
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
```

model 是 DiffusionTrainingModule (nn.Module), named_parameters 递归展开含
`pipe.dit.blocks.N.attn.sla_module.proj_l.*`, 子串匹配 "sla_module.proj_l" 正确。
无 proj_lr 时走原路径, 不影响其他训练。

### 阶段一训练脚本 (MiniMax-H3-T2VA-SLA-stage1-projonly.sh)

从 stage2.sh sed 复制改: LR=1e-4, `--proj-only --teacher-align --align-timesteps 50`,
`--task "sft:train_align"`, `--dataset_repeat 2` (2000×2÷16=250 步, 接近 200 步目标),
OUT→`MiniMax-H3-T2VA-SLA-stage1-projonly`。零初始化 proj_l 不注入。
阶段二: 用 stage1 产物作起点, 去掉 --proj-only, 加 --proj-lr 5e-5, 主干 lr 1e-5,
task 用 sft:train (普通 loss) 或 sft:train_align 均可 (用户倾向普通 SFT loss)。

## ⚠️ SLA 单卡推理: 用 diffsynth pipeline 不用 vllm serve

用户要"50 步推理一版"时, 正确做法是 diffsynth pipeline 单卡流式加载 + CPU offload
(tools/infer_sla_stage2_cafe.py 模式), **不要起 vllm serve**:

- vllm serve 8 worker 各从共享盘读 text_encoder 63GB + transformer 66GB ≈ 129GB×8,
  加载 10-30 分钟 (共享盘 IO 瓶颈), 单文件 66GB 更慢 (日志 "Multi-thread loading
  shards: 0/1" 串行读); 单条推理完全没必要付这个成本。
- diffsynth 单卡: 流式加载逐层搬, CPU offload 参数放内存, 50 步 ~8-11 分钟出片,
  和 cafe_base 之前快的原因一致 (之前也是 diffsynth 不是 vllm!)。
- SLA 推理必须给 transformer ModelConfig 传 extra_kwargs:
  `ModelConfig(path=[CKPT], extra_kwargs=dict(use_sla=True, sla_topk=0.05,
  sla_feature_map="softmax", sla_blkq=64, sla_blkk=64), **cpu_vram)` —
  与训练时 train.py:45-57 的注入方式一致。加载后确认
  `pipe.dit.use_sla == True` 和 ckpt 含 `blocks.N.attn.sla_module.proj_l.*`
  (SLA ckpt 635 keys vs dense 535 keys, 差 100 = 50 blocks × proj_l 2 键)。
- 推理容器: .28 的 h3_train_hang **不能跑 vllm serve** (glibc "corrupted size
  vs. prev_size" → orchestrator init 失败 exit 134, 之前跑通用的是 vllm-omni-h3-28
  容器); 但 diffsynth 单卡推理在 h3_train_hang 没问题。

## vllm serve 加载慢的另一个修法: 单文件拆 13 分片

若确需 vllm serve (批量/多人共享): 训练产物是单文件 step-N.safetensors (66GB),
vllm 串行读极慢。拆成 13 分片 (tools/split_ckpt_13shards.py): 按 base
FL2VA/transformer/model.safetensors.index.json 的 weight_map 把 535 keys 分回
model-00001..13-of-00013.safetensors, 重写 index.json 指向分片。
实测 13 分片并行加载 ~2.5 分钟到 85% (vs 单文件 9 分钟才读一半)。
目录结构必须 `<ROOT>/FL2VA` (pipeline 按目录名解析 model_root), 组件
text_encoder/vae 用 `cp -al` 硬链接复用。

## 经验: 用户目检是最终裁判

- SLA v1 loss 看似"在降" (0.77→0.52) 但用户目检"噪声很大" — loss 震荡
  (0.006↔0.91) 才是真相, 均值曲线会骗人。交付推理产物时主动拉 tensorboard
  分窗 + 对比 dense 同数据同步数的 loss 曲线。
- 用户对训练方案有明确技术偏好 (两阶段: 先 proj-only 补偿再联合), 实现前先
  确认 loss 类型 (普通 SFT vs teacher-align), 别默认沿用脚本现状。

## ⚠️ TeacherAlign loss 的 deepspeed 断言坑 (stage1 启动即崩, 2026-08-17 修复)

- **现象**: stage1 (proj-only + teacher-align) 启动后第一步就崩:
  `AssertionError: loss must be a scalar tensor` — deepspeed engine.py:3166
  `maybe_loss_for_backward` 要求 `value.numel()==1 and value.grad_fn is not None`。
- **根因**: TeacherAlignMiniMaxH3AudioVideoLoss (loss.py:103) 原实现返回
  `torch.stack(losses).mean().detach()` — 梯度已在每层 sla forward 内
  backward 累加 (sla_core.py:241 `loss_i.backward()`, 逐层反传设计), 返回
  detach 值只作日志; 但 deepspeed 的 accelerator.backward(loss) 要求 loss
  带 grad_fn, detach 后断言失败。
- **修复 (loss.py:152)**: 返回带 grad_fn 的标量, 且本处 backward 梯度为 0,
  不干扰层内已累加的梯度:
  ```python
  align_mean = torch.stack(losses).mean()   # detach 值的 mean, 无 grad_fn
  zero_grad_anchor = sum(p.sum() * 0.0 for p in pipe.dit.parameters() if p.requires_grad)
  return align_mean + zero_grad_anchor      # grad_fn=True, backward 梯度全 0
  ```
  容器内验证: `total.grad_fn is not None` + backward 后参数梯度全 0 → PASS。
- **可复用教训**: 自研 loss 若"梯度已在别处 (层内) 累加、返回只作日志", 必须
  给返回标量挂 0 梯度 anchor 满足 deepspeed 断言, 不能直接 detach 返回。
  同类报错先查 deepspeed 的 maybe_loss_for_backward 源码 (engine.py), 确认
  断言条件再改 loss 返回。

## ⚠️ zero3 冻结混合 dtype bug: stage1 16 卡必崩, 改单卡跑 (2026-08-17 实测)

- **现象**: TeacherAlign 断言修复后 stage1 (16 卡 zero3 + offload) 仍崩, 但错误不同:
  `AssertionError: len(set(p.ds_id for p in params_in_bucket)) == len(params_in_bucket)`
  — deepspeed/runtime/zero/stage3.py:1508 __reduce_and_partition_ipg_grads。
- **触发特征**: **bf16 主干全冻结 (requires_grad=False) + 仅 fp32 proj_l 有梯度**
  的组合。对比: stage2 全参 (bf16 主干 + fp32 proj_l 都训) 在 zero3 offload 下
  正常; 只有 stage1 (proj-only) 必崩 — deepspeed 0.19.4 对"部分参数冻结 +
  混合 dtype"的梯度 bucket 分组有 bug (ds_id 重复)。
- **解法: stage1 改单卡跑**, 绕开 zero3。proj_l 只有 3MB 参数, 单卡 + CPU
  offload 足够, 和 8/15 阶段一 (proj_l_step250.pt 单卡 250 步) 同路线。
- **单卡配置** (full/accelerate_config_single_gpu.yaml): `distributed_type: 'NO'`,
  `num_processes: 1`, 无 deepspeed_config; 脚本加 `--enable_model_cpu_offload`
  (train.py:307 device=cpu), 去掉 --main_process_ip/port。
- **多任务同机并行**: .28 上同时跑推理 (npu0) + stage1 (npu1) 时, 脚本里
  `export ASCEND_RT_VISIBLE_DEVICES=1` 隔离; 推理脚本默认 set_device(0)。
- stage1 单卡脚本: full/MiniMax-H3-T2VA-SLA-stage1-projonly-1gpu.sh
  (ASCEND_RT_VISIBLE_DEVICES=1 + single_gpu config + enable_model_cpu_offload)。
- **注意**: .4 的 stage2 v2 训练没绑 ASCEND_RT_VISIBLE_DEVICES (用全部 16 卡),
  所以 .4 上无法腾卡做推理/stage1 — 用户指定推理和 stage1 都放 .28。

## ⚠️ train.py 单卡模式必崩: meta tensor 坑 (2026-08-17 晚实测)

- **现象**: stage1 单卡脚本 (accelerate_config_single_gpu.yaml + `--enable_model_cpu_offload`)
  启动即崩: `NotImplementedError: Cannot copy out of meta tensor; no data!`
  (torch_npu/utils/_module.py:76 convert, 在 module._apply 时触发)。
- **根因**: train.py 的 enable_model_cpu_offload 走 OffloadTrainingManager, 但传给
  from_pretrained 的 ModelConfig **没有 offload 配置** (parse_vram_config 默认分支
  返回 `{}`) → 模型以 meta tensor 初始化 (延迟加载路径), 单卡非 deepspeed 模式
  没有 zero3_init 把 meta 填成实际权重 → _apply 拷贝 meta 崩。
- **结论: train.py 框架不支持单卡非 deepspeed 训练** (meta init 依赖 deepspeed)。
  单卡 stage1 必须写**独立脚本**, 复用推理脚本已验证的加载路径:
  `tools/train_stage1_projonly_1gpu.py` — ModelConfig(**cpu_vram)
  (offload_device="cpu", 推理脚本同款, 无 meta), 手动做:
  ① 冻结主干只留 proj_l (`"sla_module.proj_l" in name` → requires_grad, 其余 False);
  ② `os.environ["SLA_PROJ_ONLY"]="1"` + `os.environ["SLA_ALIGN_TEACHER"]="1"`;
  ③ 50 点网格 `scheduler.set_timesteps(50, training=True, shift=12.0)` (teacher align
  训练分布=推理分布, 和 train.py teacher_align 模式一致);
  ④ 手动组装 inputs (缓存 .pth 的 shared+embeds 合并, tensor `.to(DEVICE)` + bf16,
  模仿 train.py transfer_data_to_device); use_gradient_checkpointing 强制 False
  (teacher align 逐层 backward 不能叠加 checkpoint 重算);
  ⑤ TeacherAlignMiniMaxH3AudioVideoLoss + AdamW(只含 proj_params, lr 1e-4) +
  每步 empty_cache; ⑥ 只导出 proj_l 存 `.pt` (主干零漂移 = 原始 FL2VA, 阶段二
  直接用原始主干 + stage1 proj_l)。
  数据用 200 条子集 (`ir5k_gen_latent_cache_480p_ckpt_200`, 软链接前 200 个 pth)
  = 200 步 (单卡 batch 1)。
- ⚠️ **该独立脚本 2026-08-17 晚写完, 后续会话已跑通到第一步** — 模型加载
  (4s, CPU offload 流式) + 冻结生效 (100 个 proj_l 对象 / 825600 params) + 50 点
  网格 (sigma[0]=1.0, sigma[-1]=0.1967) 全部正常, 第一步 forward/backward 执行中
  (单卡逐层对齐, 首步含编译预热 3-5 分钟)。跑通过程又踩了两个新坑 (见下)。

### ⚠️ 独立脚本两个必踩坑 (2026-08-17 晚跑通时实锤)

**坑 1: proj_l 是 VRAM offload 的 meta 占位 → 必须显式 materialize。**
现象: 启动后 sla_core.py:224 `self.proj_l(o_l...)` 报
`NotImplementedError: Cannot copy out of meta tensor; no data!`
(vram/layers.py:62 cast_to → r.copy_(weight))。
根因: 原始 FL2VA 权重文件里**没有 proj_l** (SLA 新增参数), from_pretrained 的
VRAM offload 包装把 proj_l 变成 meta 占位延迟加载, 但没有任何文件能填它。
修复 (加载后冻结循环之后):
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
零初始化与 sla_core 默认一致 (nn.init.zeros_)。替换为普通 Parameter 后不再走
VRAM cast 路径, 直接可用 (proj_l 才 128×128, 无需 offload)。

**坑 2: inputs 含嵌套 dict (packed), 转 device 必须递归。**
现象: 修完坑 1 后报
`index_copy_ error: index (img_pos) 在 cpu, 其他 tensor 在 npu:0` —
model_fn 里 `x[0].index_copy_(0, img_pos[...], video_rows)`。
根因: 缓存 .pth 的 embeds["packed"] 是嵌套 dict, 单层循环
`for k, v in inputs.items()` 只转了顶层 tensor, packed 里的
img_pos/audio_pos/text_pos 等留在 cpu。
修复: 递归 _to_device (与 train.py transfer_data_to_device 同语义):
```python
def _to_device(v):
    if isinstance(v, torch.Tensor):
        t = v.to(DEVICE)
        if v.dtype in (torch.float, torch.float16, torch.bfloat16):
            t = t.to(dtype)
        return t
    elif isinstance(v, dict):
        return {k: _to_device(x) for k, x in v.items()}
    elif isinstance(v, (list, tuple)):
        return type(v)(_to_device(x) for x in v)
    return v
inputs = _to_device(inputs)
```
⚠️ 附带风险: packed 的 img_position_ids 是 float64, 转 npu 后 NPU 不支持
double (日志见过 "Device do not support double dtype"), 若该字段报错需
额外 `.to(torch.float32)` — 本次跑到第一步尚未触发, 遇报错再处理。

### stage1 完成后自动接 stage2 (用户流程要求)

用户明确: "stage1 完事了赶紧接 stage2" — 两阶段是串行自动接力, 不是等人确认。
stage2 启动配置: 用 stage1 产出的 proj_l (stage1 只存 proj_l .pt, 主干零漂移 =
原始 FL2VA), 去掉 --proj-only, 加 --proj-lr 5e-5, 主干 --learning-rate 1e-5,
task 用 sft:train (普通 SFT loss, 用户 8/17 已定)。

## ⚠️⚠️ 单卡 teacher 对齐必须用"两遍法", 不能用 SLA_ALIGN_TEACHER 逐层反传 (2026-08-17 晚实测, 最重要教训)

- **现象**: 上面独立脚本 (SLA_ALIGN_TEACHER=1, TeacherAlignMiniMaxH3AudioVideoLoss
  在 sla_core forward 内逐层立即 backward) 单卡跑, **第一步 35 分钟未完成**:
  AICore 42% 不涨 (NPU 没吃饱), 主线程 sleep 等 NPU, syscr 117 万次 read 调用 =
  CPU offload 状态机在反复搬运参数 (thrashing)。推理 8.4s/it 是纯 forward 流式
  搬运没问题; 训练有 backward 要参数留在卡上, offload 管理不同 → 抖动。
- **解法: 复用 8/15 阶段一脚本 `tools/train_proj_only.py` 的两遍法** (当时 250 步
  跑通, 产物 proj_l_step250.pt):
  1. **第一遍 forward 全程 `torch.no_grad()`** (显存 = 推理级, 10.9GB):
     每层 sla_module 挂 `register_forward_pre_hook` 抓 o_full (全注意力 teacher),
     `register_forward_hook` 抓 o_sla 和 o_l (SLA_PROJ_ONLY 时 sla_core 存
     `self._o_l` = 原始线性输出), 全部 `.cpu()` 存列表。
  2. **第二遍梯度模式只对 proj_l 分支重算**: 对每个 (o_s, o_l, o_full):
     `o_sla = o_s + proj(o_l.to(fp32)).to(dtype)`, `F.mse_loss(o_sla, o_full)`,
     逐层独立小图 `loss_i.backward()` 只更新该层 proj_l, 图用完即释放。
- **实测速度**: step1=135s (含加载), step10=303s, step20=501s (~20s/步),
  200 步 ≈ 1.1-1.8h; loss 27→21 下降, proj|w| 0.0001→0.0011 增长 (学习正常)。
  对比 SLA_ALIGN_TEACHER 单卡 35min/步没跑完 — **两遍法快 15 倍以上**。
- **判断规则**: SLA_ALIGN_TEACHER 逐层立即 backward 是为 16 卡 zero3 设计的;
  单卡 + CPU offload 下参数搬运 thrash, 不可用。单卡训 proj_l 一律两遍法。
- 复用命令: `cp tools/train_proj_only.py tools/train_proj_only_200.py`, 改
  CACHE 为 200 条平铺缓存 + glob 改 `CACHE/*.pth` (8/15 原版是 `CACHE/*/*.pth`
  分片结构), 跑 `python3 tools/train_proj_only_200.py --steps 200 --lr 1e-4
  --out .../proj_l_align_200 --save-every 50`。
- 缓存格式验证: 平铺 .pth 与 split-cache 同为 (shared, posi, third) 3 元组 —
  input_latents/audio_input_latents/imgvid_cond_noise_aug 在 shared,
  prompt_embeds/packed (img_pos/audio_pos/text_pos/img_position_ids/token_tags/
  cu_seqlens/seq_len) 在 posi。200 条子集 = `ir5k_gen_latent_cache_480p_ckpt_200`
  (软链接前 200 个, 单卡 batch 1 正好 200 步)。

## ⚠️ c10d 初始化 gai error (2026-08-17 晚实测, 单机多卡重启必备)

- **现象**: 训练重启 (offload 版) 卡死 30 分钟, 日志反复
  `IPv6 network addresses of (worker-97-4, 45529) cannot be retrieved (gai error: -3
  Temporary failure in name resolution)` — accelerate 用主机名做 c10d 初始化,
  而 /etc/hosts 里只有 `master-node-97-4` 没有 `worker-97-4`。
- **诱因**: 上一个训练进程残留占着 29500 端口 → accelerate 自动换端口, 叠加
  主机名解析失败, 通信组初始化混乱。杀残留后端口释放, 但主机名问题仍在。
- **修复 (双保险)**: ① launch 加 `--main_process_ip 127.0.0.1 --main_process_port 29500`
  (单机多卡标准做法, 绕开主机名解析); ② /etc/hosts 加别名
  `90.90.97.4 worker-97-4 master-node-97-4`。
- **可复用**: 单机多卡重启前先 `ps -eo pid,args | grep [a]ccelerate` 杀残留 +
  `ss -tlnp | grep 29500` 确认端口干净, 再启动; 别让残留进程占端口。

## ✅ stage1 完整跑通 (2026-08-18 凌晨实锤) + stage2 接力配方

**stage1 两遍法 200 步全部跑完**: 3842s (64 分钟, ~19s/步), 产物
`proj_l_align_200/proj_l_step200.pt` (100 keys = 50 blocks × weight/bias,
**mean|w| = 0.0049**, 和 8/15 的 proj_l_step250 的 0.0053 同量级 → 训练有效)。

**⚠️ 健康信号判据: 看 proj|w| 单调增长, 不是 loss 下降。**
stage1 全程 loss 剧烈震荡 (27→21→18.9→24→13.9→25→23.7, 步 10 采样就 27↔28),
但 proj|w| 从 0.0001 → 0.0062 **单调增长从未回落** — 这就是 proj_l 在学的铁证。
loss 震荡是逐层对齐 MSE 的固有噪声 (各层/各 timestep 差异大), 均值没意义;
用户若质疑"loss 没降", 直接报 proj|w| 曲线。

**stage2 接力配方 (实测跑通):**
1. `.pt` → `.safetensors`: `safetensors.torch.save_file(torch.load(pt), out)` (100 keys)。
2. stage2.sh: `PROJ_ST=proj_l_align_200/proj_l_step200.safetensors`,
   MODEL_PATHS_DIT 恢复 `[tr + [sys.argv[2]]]` (13 shards + proj 同包一个
   ModelConfig, 两阶段法的 hash 配置), `--proj-lr 5e-5` 加在 --learning_rate 后,
   LR=1e-5 (主干), OUT 换新名 (`SLA-stage2-step200-2stage`)。
3. 验证 12 断言 (bash -n + PROJ_ST 路径 + 双 lr + offload + 分布式修复 + SLA
   配置 + sft:train + 2000 条缓存) 后 .4 16 卡启动。
4. 预期: 主干 1e-5 + proj_l 5e-5 双 lr 分组打印 `[runner] 双 lr: 主干 ... proj_l ...`,
   125 步 (2000÷16), save_steps 50 → step-50/100/125。

**v2 (proj0+lr1e5) 全量完成参考**: 125 步 3:00:32 (86.66s/it), step-50/100/125
三档各 66GB; 训练进程退出后最终档 (step-125) 保存还要 ~5 分钟 (写共享盘),
判断"训练完"以 `ls` 见 step-125 + accelerate 进程数归零为准, 别以日志 125/125 为准。

## ⚠️ 自动接力 watcher 模式 (2026-08-18, 用户要求"训完自动推理/续训")

用户偏好端到端自动化: "跑完250步自动推理"、"125跑完也续训到250,然后推理一版"。实现模式:

- **watcher 脚本** (host 域跑, 容器内无 docker CLI — 见 memory): 轮询 `step-N.safetensors`
  出现 → **大小稳定检查** (`stat -c %s` 每 30s 轮询, 连续两次相同且 >66GB 才认为写完,
  66GB 写共享盘要几分钟, 日志 125/125 出现 ≠ 训练完) → 执行推理/续训命令。
- **等训练进程退出再续训**: `while docker exec ... grep -c '[a]ccelerate launch' | grep -q "[1-9]"; do sleep 60; done`
  (注意 `[a]ccelerate` 方括号防自匹配, 验证脚本若查 `"accelerate launch" in s2` 会误报 FAIL)。
- 启动: `nohup bash watch_x.sh > .../watch_x.log 2>&1 &`, 日志落共享盘便于排查。
- 续训脚本 = 原训练脚本复制改三点 (MODEL_PATHS_DIT 软链加载 step-NN + 恢复原配置 + OUT 换新名),
  见下节"续训配方" — **勿用 --resume_from_checkpoint** (zero3 offload 下必崩 shape[0], 实测)。
- ⚠️ **watcher 轮询的是文件路径不是进程**: 若续训脚本崩了 watcher 不会死 — 修好脚本
  手动重启训练即可, watcher 继续轮询自动接力 (本会话 2stage watcher 实测: 自动启动的
  旧 resume 版崩了, 改 MODEL_PATHS_DIT 版手动重启后 watcher 正常接管)。别杀 watcher。
- ⚠️ **两阶段法续训必须保留 --proj-lr 5e-5 (双 lr), v2 续训才是单 lr** — "恢复原配置"
  按版本区分; 从 v2 脚本 sed 复制成 2stage 版时容易把 --proj-lr 一起删掉, 记得加回。
- 推理产物命名带版本后缀 (step50/step125/cont250) 防覆盖; 用户"50步照例推一版"节奏 —
  中途档也推, 对比收敛效果。

## 用户监视命令偏好 (反复触发的教训)

- 用户抱怨过 "你这个监视命令根本跑不通" (嵌套引号 bash -lc 失败)。**给用户的
  监视命令必须单层引号**, 或写成 bash 脚本放共享盘 `tools/` 让用户跑一条简单
  命令: `bash /mnt/a800_share/c50058431/tools/<name>_monitor.sh` (模式:
  while true + clear + 显示进程/卡占用 + sleep 15, Ctrl+C 退出不影响任务)。
- 示例: serve_monitor.sh (vllm 加载进度)、stage1_monitor.sh (训练进程+卡占用)。
- 训练进度/日志在 Hermes 后台进程时, 脚本 stdout 用户看不到 — 训练脚本应
  tee 到共享盘日志文件, 或用户直接用 tail 命令盯日志。

## ⚠️⚠️ 续训 (resume) 配方 — 2026-08-18 实测: --resume_from_checkpoint 在 zero3 offload 下必崩, 改用 MODEL_PATHS_DIT 直接加载

用户要求 "在 .28 机器上续训之前跑完的 125 步, 往后续到 250 步看结果"。

### ❌ 失败的路径: --resume_from_checkpoint (zero3 offload 下必崩)

- **现象**: 加 `--resume_from_checkpoint $OUT/step-125.safetensors` 启动, rank0 崩:
  `RuntimeError: Error(s) in loading state_dict for MiniMaxH3TrainingModule:`
  `size mismatch for pipe.dit.video_patch_proj.weight: copying a param with shape`
  `torch.Size([5376, 96]) from checkpoint, the shape in current model is torch.Size([0])`
  (每个主干参数都报 [0], 包括 time_embedder/rope/token_refiner)。
- **根因**: train.py:70 的 `resume_from_checkpoint` 在 `switch_pipe_to_training_mode`
  (72 行) 和 accelerator.prepare **之前**调用; zero3 offload 下 from_pretrained 后
  参数还是 meta/占位 (shape [0]), 此时 load_state_dict 拿 66GB 真权重去填必然
  size mismatch。原 v2 训练没 resume 所以没暴露。
- **结论: zero3 offload 模式下不要用 --resume_from_checkpoint 续训。**

### ✅ 正确的续训路径: MODEL_PATHS_DIT 直接指向完整 ckpt (from_pretrained 加载)

原理: step-125.safetensors 是全量权重 (含 proj_l, 635 keys), 直接当 transformer
起点加载 — 与推理脚本加载方式、stage2-step200-2stage 的 proj 注入方式同构,
都有实测跑通先例。步骤:

1. **软链 + "transformer" 字样陷阱**: train.py:45-57 的 use_sla extra_kwargs 注入
   逻辑检查 `"transformer" in str(model_config.path)` — step-125.safetensors 路径
   不含 "transformer" → use_sla 不注入 → dit 按 dense 构造 → 加载含 proj_l 的
   ckpt key 不匹配。**解法**: 在输出目录建含 "transformer" 的软链:
   ```bash
   mkdir -p $OUT_ROOT/MiniMax-H3-T2VA-SLA-v2-cont250
   ln -sf $OUT_ROOT/<原训练目录>/step-125.safetensors \
          $OUT_ROOT/MiniMax-H3-T2VA-SLA-v2-cont250/transformer_step125.safetensors
   ```
2. MODEL_PATHS_DIT 生成改成指向该软链 (单文件):
   ```python
   p = os.path.join(sys.argv[1], "MiniMax-H3-T2VA-SLA-v2-cont250", "transformer_step125.safetensors")
   print(json.dumps([[p]]))   # 注意单文件也要包成 [ [path] ] 内层列表
   ```
3. **不要传 --resume_from_checkpoint** (删掉), --remove_prefix_in_ckpt "pipe.dit."
   保留 (保存时去前缀, 加载时 ckpt key 本来无前缀, 无碍)。
4. 其余同原训练配置: 单 lr 1e-5 (v2) 或双 lr (两阶段法续训保留 --proj-lr 5e-5)、
   OUT 换新名、task sft:train、2000 条缓存、offload + 分布式修复参数。
5. 验证要点 (11 断言): bash -n + 无 --resume_from_checkpoint + 软链存在 +
   路径含 "transformer" (use_sla 命中) + LR + OUT + SLA 配置。注意验证脚本
   查 "无 --resume_from_checkpoint" 时排除注释行 (注释里会提这个词)。
6. 单文件 66GB 加载较慢 (zero3 每 rank 读), 等进度条出现再判健康; 内部计数
   从 0 重计 (续训 125 步 = 累计 250 步), 汇报必须说明累计口径 (用户偏好)。
   实测确认 (v2-cont250, 2026-08-18): 软链加载成功, 98s/it (略慢于首次 85s —
   单文件 + 参数稍多), 稳定推进到 125/125; 完成后最终档同样叫 step-125.safetensors
   (续训目录内, 累计 250 步), watcher 轮询该路径即可自动接推理。

- **口径**: 续训内部计数从 0 重计 (续训 125 步 = 累计 250 步), 汇报必须说明
  累计口径 (用户既定偏好)。

## FAQ: "每步 16 条更新一轮权重, 8 条是不是也够了" (2026-08-18 用户提问)

- **现状**: 16 卡 zero3, 每卡 batch 1 → 每步 = 16 条样本梯度平均后更新一次权重。
- **为什么是 16 卡**: 33B 模型单卡 64GB 装不下 (66GB 权重 + 激活), 必须 zero3
  分片到多卡; 卡数定了, 每卡只能放 1 条 (激活显存限制) → 每步 16 条是"被迫"的,
  不是特意选的大 batch。**16 条/步对 33B 其实很小** (行业惯例 batch 256-4096),
  8 条只会梯度更抖、收敛更不稳 — 8 条不够, 不是够。
- **8 条的代价**: ①用 8 卡训 → 每步 8 条, 数据过一遍要 2 倍步数, 且 zero3 8 卡
  每卡参数分片更大显存更紧; ②保持 16 卡但每卡 batch 减半 → 浪费显存无意义。
- **用户真正想要"更新更稳/数据用更慢"时**: 正确方向是 **gradient accumulation**
  (如 2 步累积再更新 = 等效 batch 32), 步数/数据量/时间不变, 只降更新频率;
  或 `--num_epochs` 加大多过几遍数据。回答这类问题先说机制 (卡数=分片必需,
  每步条数=卡数×每卡 batch), 再给选项, 别默认用户要改 batch。

## ⚠️ 口径: 总步数 vs 中途档 (用户两次问"不是50步的吗"的教训)

2000 条 ÷ 16 卡 = **125 步总训练** (dataset_repeat 1); save_steps=50 只是自动存
中途档 (step-50/100/125)。用户会把"推了 step-50 档"记成"总共只训了 50 步" —
**汇报时必须说明: 总步数 (125) + 当前档位 (step-50 = 40% 处, step-125 = 最终档)**。
dense 的"50 步"是 800 条 ÷ 16 卡; 推理 ckpt 名 (step-50) 是档位不是总步数。

## ⚠️ 并行推理: 多卡设备隔离 + 三要素核对 (用户两次纠正的教训)

- 两路推理并行: `sed 's/torch.npu.set_device(0)/torch.npu.set_device(1)/' <源脚本>`
  复制出绑第二张卡的版本 (npu0 跑一路, npu1 跑另一路), 输出文件名同步改。
- ⚠️ **同机两路 diffsynth 推理并行会互相拖慢**: 实测 50 步从 8s/it 涨到
  14.9-20.9s/it (CPU offload + 共享盘 IO 争抢), 出片 20 分钟 vs 单路 11 分钟。
  赶时间就串行; 并行则接受 ~2x 慢, 别误判为卡死。
- **教训 (本会话用户两次纠正)**: 从旧脚本 sed 复制时, 必须逐项核对
  **CKPT 路径 / set_device 卡号 / 输出文件名 三要素**。曾从 step-125 脚本复制
  只改 device 就启动, 结果 npu1 重复跑了 step-125 (覆盖同名输出), 用户实际
  要的是 stage2 的 step-50 — 用户原话 "npu1上的不是stage2 50步的吗"。
- 训练版本多时, 用户口中的"那个训练"可能指任一历史版本 (SLA-3600-proj 450 步 /
  v2 125 步 / 两阶段法 step200-2stage) — 启动推理前若不确定 ckpt, 先
  `ls models/train/` 列目录+时间戳与用户对齐, 输出文件名带版本后缀
  (如 sla_stage2_proj0_lr1e5_step125_cafe.mp4) 防覆盖。
- **推理跑哪台机器: 优先训练刚完成的那台 (用户纠正 "续训完推理直接在.4推呀", 2026-08-18)**。
  续训在 .4 完成后 .4 的 16 卡全空, 推理直接在 .4 npu0 跑 (diffsynth 单卡);
  别默认放 .28 — .28 正 16 卡跑 latent 采集, 推理挤进去会抢卡拖慢采集
  (采集是 rank 分片常驻进程, 每 rank 绑一张卡)。若推理不得不与采集同机,
  先暂停对应 rank (杀该 rank 进程, 断点续跑自动跳过已存在样本, 最多损失
  当前一个 ~7min 样本), 推理完再按原命令重启该 rank。watcher 脚本里推理
  命令直接写 `docker exec h3_train_hang ...` (本地) 而不是 `ssh .28`。
- 训练完成判定补充: 日志 125/125 出现 ≠ 训练完, 最终档保存还要几分钟
  (66GB 写共享盘), 以 `ls` 见 step-125 + accelerate 进程数归零为准;
  中途档 step-50 也值得推 (用户"50步照例推一版"节奏), 对比 step-50 vs step-125
  看收敛效果。

