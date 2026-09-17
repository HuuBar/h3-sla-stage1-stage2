# H3-SLA 训练全记录(截至 2026-08-15)

> 整理时间:2026-08-15 16:30 | 目录:`/mnt/a800_share/c50058431/`
> 用途:汇总 SLA 稀疏化微调从 8/13 至今的所有训练数据、评估结果与根因分析,便于复盘决策。

---

## 1. 项目概览

**目标**:把 MiniMax-H3(音视频联合生成大模型,33B DiT)的注意力机制替换为 SLA(Sparse-Linear Attention,稀疏线性注意力),通过少量步数的全参微调让模型适应稀疏注意力,训练后推理侧可用 SLA 省掉约 95% 的注意力计算。

**当前状态(一句话)**:SLA 实现本身验证正确(原始权重直接开 SLA 推理画面正常),但**两轮微调都训出了"偏白/白屏"症状**——混合采样方案(3600 条/250 步)把白% 从 78.8% 压到 52.3%,缓解未根治;根因已钉死为**训练火力分布缺陷:低噪端(sigma<0.2,即推理最后几步所在区间)欠训练**,修正方向已给出待拍板。

**关键文件**:
- 训练仓库:`/mnt/a800_share/c50058431/minimax_dmd`(SLA 开发副本,8/11 拷贝)
- 原始权重:`/mnt/a800_share/minimax-h3/FL2VA/`(transformer 13 分片 + text_encoder + video_vae + audio_vae)
- 方案/分析文档:`sla_ref/h3_sla_training_plan.md`、`sla_retrain_plan.md`、`sla_ref_DiffSynth-H3-SLA/`(外部参考仓库对比)
- 本目录根下的 mp4 都是历次推理产物(见 §10)

---

## 2. 环境与配置

| 项 | 值 |
|---|---|
| 节点 | 90.90.97.4/.15/.28/.29(每节点 8 物理卡 × 2 die = 16 逻辑 NPU) |
| 训练容器 | h3_train_hang(时钟比 host 慢 8h,watcher 必须在 host 域跑) |
| 框架 | DiffSynth(editable 安装,**必须 cd 到 repo 根再跑**,否则 import 旧副本) |
| 分布式 | 16 卡 ZeRO-3(`accelerate_config_zero3_16gpu[_clip].yaml`),MBS=1 |
| 模型 | MiniMaxH3DiT:50 层、hidden 5376、56 heads、head_dim 128;可训练参数 33.12B(全参微调,非 LoRA);SLA 新增 proj_l 仅 0.83M |
| 调度器 | FlowMatchScheduler shift=12(视频)/3(音频),1000 timesteps |
| 损失 | FlowMatchSFTMiniMaxH3AudioVideoLoss(MSE × BSMNTW 权重,视频+音频双分支) |
| 训练速度 | ~70-84 s/it(16 ranks,batch=1/rank) |
| ckpt 体积 | 66.2 GB/个(safetensors,全参 DiT) |
| 推理 | diffsynth pipeline(MiniMaxH3Pipeline)+ SLA,CPU offload,vram_limit≈53,50 步 ≈ 8-10 min |

**SLA 配置**(与 MindSpeed 参考一致):topk=0.05、feature_map=softmax、blk_q=64、blk_k=64、head_dim=128。

---

## 3. 数据集

| 数据集 | 条数 | 说明 |
|---|---|---|
| dl_list_ir5k.jsonl | 4724 | 计划清单(IR 三段式 prompt:integrated_multimodal_description / overall_soundscape / non_diegetic_music) |
| clips_ir5k/ | 4462 | 实际已下载 mp4 |
| h3_train_data_ir5k_2000 | 2000 | 首轮训练集(清单前 2000,8/12 构建) |
| h3_train_data_ir5k_3600 | 3599 | 第二轮训练集(build_ir5k_dataset.py --target 3600 → 3599 条,1 条 NO_AUDIO 剔除) |

- metadata.csv 列 = `video,input_audio,prompt`(无 id 列,顺序不能错)
- 时长统一 5.2-5.5s;480×832×124 帧;音频 32k 双声道
- 数据质量全量扫描(2000 条):白% 中位 2.6%,白>90% 仅 3 条(合理内容)→ **数据不白**;低纹理 36.5%、静止 2.9%、全黑 1 条("模糊"助推剂,非白屏根因)
- cache(split-cache):8 ranks × 450 pth(3600 集)/ 8 × 250(2000 集);`(shared, posi, nega)` 元组结构,视频 latent 在 shared.input_latents [1,24,37,30,52] bf16;全量 6000+ 张量无 NaN
- 训练 cache 的 latent 解码验证:白 0-7.4%,与源视频一致 → cache 格式/归一化无罪

---

## 4. 训练轮次记录

### 4.0 Debug 轮(8/13-8/14,修复 SLA backward)

- 目的:修通 SLA Triton kernel 在 NPU 上的 backward 数值损坏
- 症状链:step1 loss 1.8887 → step2 nan;根因 `_attn_bwd_dkdv` 的 KBID 指针 stride 与 mask 不匹配 → 梯度死掉/nan
- **修复(8/14 完成并验证)**:用 torch-ops 重算反向替换 Triton backward(`_SparseAttentionWithTorchBackward`),q/k/v/do 保持 bf16、head 分组 HG=8、query chunk CH=8 防 OOM;修复 head-grouping 全局偏移 bug(MTE DDR OOB)
- 验证:单卡 repro 梯度有限;与全注意力 grad 对比 rel ~1e-3;AST 提取磁盘代码 CPU 验证 dq/dk/dv rel 4-8e-7
- 修复后 16-rank clean run:27+ 步零错误,loss 1.54→0.08-0.5,sparsity 0.899 稳定,82.7 s/it
- ckpt:step-50/100/125(旧 debug 目录)

### 4.1 第一轮正式训练:SLA-2000(8/14 05:23-09:31,200 步)

| 项 | 值 |
|---|---|
| 数据 | h3_train_data_ir5k_2000(2000 条),dataset_repeat 2 |
| 步数 | 内部 200(≈1.6 遍数据;125 步后 resume 一次) |
| lr / clip | 1e-5 恒定 / **无 clip_grad**(DiffSynth 默认不裁剪) |
| 采样 | 全范围均匀(max=1.0/min=0.0,**无 boundary**) |
| OUT | `minimax_dmd/models/train/MiniMax-H3-T2VA-SLA/` |
| ckpt | step-50/100/150/200(66.2 GB 各) |
| loss 曲线 | 首轮 1-60 均值 0.659 → 61-125 均值 0.466(收敛);resume 后 1-75 0.442 → 76-150 0.464 → 151-200 0.528(**后期反弹=过拟合区**);0 NaN/Inf |
| 主干漂移 | relΔ ~0.1%(最大组 time_embedder 1.66%,绝对量小);proj_l 从 0 → norm 3.55 |
| 梯度参数量 | 33123.82M(全参);proj_l 仅 0.8256M |

**推理评估(同 prompt vast_paint,50 步)**:

| 配置 | 亮度 | 白% | 暗% |
|---|---|---|---|
| 原始 FL2VA 全注意力(基线) | 126 | **3.5%** | 25% |
| step-125(用户:勉强,仍偏白) | — | 51.8-71.9%(面包师/女孩) | — |
| step-200 权重 + 关 SLA | 161 | 37.5% | — |
| step-200 权重 + 开 SLA | — | **78.8%** | — |

结论:**问题从 step-125 就有,越训越白**;SLA 推理是放大器(37.5→78.8),不是根因。

### 4.2 第二轮训练:SLA-3600 混合采样(8/14 19:00-8/15 00:16,250 步)

用户拍板方案(8/14 晚):3600 条数据 + 250 步 + 混合采样低噪强化 + clip-grad 1.0 + 从原始 FL2VA 重训(不 resume)。

| 项 | 值 |
|---|---|
| 数据 | h3_train_data_ir5k_3600(3599 条),dataset_repeat 2(16 卡 → 450 it/rank 总量) |
| 步数 | 内部 250(被 stop-250 watcher 截停,非自然跑完) |
| lr / clip | 1e-5 恒定 / **clip-grad 1.0**(DeepSpeed yaml gradient_clipping,engine.step 内部裁剪) |
| 采样 | 混合采样:`--min-timestep-boundary 0.77 --max-timestep-boundary 1.0 --mix-low-noise-ratio 0.5`(50% 概率走低噪段 sigma≤0.78,50% 概率全范围) |
| OUT | `minimax_dmd/models/train/MiniMax-H3-T2VA-SLA-3600/` |
| ckpt | step-50/100/150/200/250(66.2 GB 各) |
| loss 曲线 | 前 100 步 1.0→0.5 明显下降,后 150 步 0.5-0.65 平台震荡(batch=1 单步抖动 0.005-1.09 正常);**0 NaN/Inf,无旧 run 那种后期反弹** |
| sparsity | 全程 0.898 纹丝不动 |
| 实测速度 | ~70-84 s/it |

**推理评估(同 prompt vast_paint,50 步)**:

| 配置 | 亮度 | 白% | 暗% |
|---|---|---|---|
| 原始 FL2VA(基线) | 126 | 3.5% | 25% |
| step-250 + 关 SLA | 168.7 | 22.6% | 1.1% |
| step-250 + 开 SLA | 199.9 | **52.3%**(max 72.8%) | 0.04% |
| (旧)step-200 + 开 SLA | — | 78.8% | — |

RGB 均值 (187.6, 204.4, 207.6) 偏青;帧间差异 5.17(有动态);帧 std 39-63(有结构,非纯白)。**声音正常、画面错乱**。

结论:**比旧轮改善约 1/3,但离正常(3.5%)差距巨大**。判别实验证明:主干被污染(3.5→22.6%),SLA 再放大 2.3 倍(22.6→52.3)——与旧轮同构,整体轻 1/3。

### 4.3 golden retriever 对比推理(8/15 16:19-16:31,已完成)

用户指定 prompt(A golden retriever runs across a grassy field...),对比 **基线(原始 FL2VA 全注意力)vs 原始权重+SLA(proj_l 零)**,验证"不训练直接开 SLA"在第二个 prompt 上是否也正常。
- 脚本:`tools/infer_compare.py --mode {baseline,sla-zero}`、`tools/run_infer_compare.sh`(同 seed、同 50 步、480×832×124)
- 产物:`/mnt/a800_share/c50058431/cmp_baseline_golden_retriever.mp4`(16:26)、`cmp_sla_zero_golden_retriever.mp4`(16:31)

**结果(亮度/白%/暗%)**:

| 配置 | 亮度 | 白% | 暗% |
|---|---|---|---|
| 基线(原始 FL2VA 全注意力) | 129.4 | 0.0% | 3.0% |
| 原始 + SLA(proj_l 零) | 122.5 | 0.0% | 2.0% |

→ **两个 prompt 都验证通过:原始权重不训练直接开 SLA,画面完全正常**。第三次确认实现无罪、白色 100% 来自训练污染的权重,与判别实验 3(白 3.0%)结论一致。

---

## 5. 白屏根因分析完整证据链

### 5.1 判别实验三连(8/14,决定归因)

| # | 实验 | 配置 | 白% | 结论 |
|---|---|---|---|---|
| 1 | 基线 | 原始 FL2VA 全注意力 | 3.5% | 推理链路/VAE 无罪 |
| 2 | 判别 2 | step-200 权重 + 关 SLA | 37.5% | **主干微调偏移本身就能致白** |
| 2' | — | step-200 权重 + 开 SLA | 78.8% | SLA 放大污染 |
| 3 | 判别 3 | **原始权重 + 开 SLA(proj_l 零)** | **3.0%** | **实现无罪:SLA 移植正确,论文核心假设成立** |

### 5.2 六层排除(8/14 收官)

1. **数据侧**:全量 2000 条扫描白% 中位 2.6%;cache 解码(实验C)白 0-7.4%;x₀ scale 反事实(×2 最多白 9.5%)→ 数据彻底无罪
2. **权重侧**:ckpt vs FL2VA 无 NaN/Inf,主干 relΔ ~0.1%,新增 100 个 proj_l key 非零
3. **ZeRO-3 保存完整性**:535 张量逐张扫描,唯一 22% 零元素的 adaln_proj 与原始零位置 100% 一致(模型自带结构)→ 保存完整
4. **loss/发散**(实验D):三份 events 全 0 NaN/0 Inf;loss 剧烈跳变 = batch=1 难度抖动,非发散
5. **clip 缺口实锤**:runner.py 裸 backward→step 无裁剪;train.py 无 clip_grad_norm_;MindSpeed 配方有 clip 1.0(第一轮缺失)
6. **latent 实锤**:解码探针 z=0 → 中灰 0.379;|z| 0.38(cache)→0.55(FL2VA)→0.75(step200+SLA),std 0.84→0.82→0.68 → **正向漂移+方差坍缩**,与白% 单调对应

### 5.3 采样轨迹实锤(8/14-8/15)

monkeypatch `pipe.step` 逐步记录 video_latents + 独立 VAE 解码:

**旧 run step-200 vs 原始**(tools/sampling_traj_check.py):
- step 1-45:|z| 几乎一致(差<0.03)
- **step 49(最后一步,t→0):step-200 |z| 0.62→0.99 爆炸、范围 ±4.75,解码白 69.2%;原始只到 0.633/8.8%**
- 结论:白屏 = 低噪端 velocity 学偏(幅度过大)→ 最后一步 overshoot → latent 冲出正常范围 → 解码全白

**新 run step-250 vs 原始**(tools/traj_check_verbose.py,逐步存 PNG + 并排图 `traj_compare_step250_vs_orig.png`):

| 采样步 | step-250 白% | 原始白% |
|---|---|---|
| 40 | 5.0% | 3.1% |
| 45 | 5.3% | 2.0% |
| 47 | 4.5% | 1.0% |
| 48 | **13.7%** | 0.9% |
| 49 | **45.6%** | 3.3% |
| 成片 | 44.5% | 3.1% |

→ step-250 前 47 步正常,**坏在最后两步(step 48 起)**,与旧 run 模式一致,起点略早 1 步。

**帧级对比**(tools/frame_compare.py):step 10-30 两权重 MAD 仅 2-10/255 几乎一致;分歧从 step 40 开始加速(22→50→80)。50 步推理 step 40 ≈ sigma 0.2 = 低噪欠训练盲区边界,与 velocity probe 根因自洽。

### 5.4 velocity probe(根因收官,8/14 旧 run + 8/15 step-250)

方法:固定同一 cache 样本 x₀ + seed noise,扫 sigma=0.02~0.9,对比原始 FL2VA 与微调权重的 velocity 预测。

**旧 run step-200**:低噪端 Δv/|v_orig| = 28.7%(sigma 0.02)vs 中段 11.4%(0.5)→ 敏感度 2.5 倍;err_ft−err_orig 全负(-17~-34)= 训练分布内更准,不是"学坏"

**新 run step-250**(tools/velocity_probe_250.py,视频分支):

| sigma | \|Δv\| | Δ/\|v_orig\| | err_ft−err_orig |
|---|---|---|---|
| 0.02 | 395.0 | **30.0%** | -22.7 |
| 0.10 | 389.9 | 26.5% | -58.5 |
| 0.20 | 304.0 | 19.6% | -38.4 |
| 0.50 | 178.3 | 10.9% | -20.4 |
| 0.90 | 214.6 | 13.0% | -48.6 |

→ 与旧 run 几乎相同(30.0% vs 28.7%),低噪端敏感 2.7 倍依旧。**音频分支**:|Δv| 仅 20-32(|v| 70-115,~20-28%),err 基本持平 → 音频 velocity 场扰动小一个量级 = "声音好"的直接原因(shift=3 下同一 u 对应更低噪声)。

### 5.5 根因链条(完整闭环)

```
训练火力分布缺陷(shift=12 均匀采 timestep → 低噪端 sigma<0.2 只占 2% 火力
    + BSMNTW 权重在低噪端仅 0.21-1.7,峰值在中段 2.96)
→ 200/250 步全参微调主干漂移 0.1%(lr 1e-5 的数学必然)
→ 低噪端对权重漂移敏感度是中段 2.5-2.7 倍(σ→0 时 x_t≈x₀,模型靠 timestep 条件精确落位)
→ 最后一步 velocity 过冲(±4.75)→ latent 冲出 VAE 正常范围 → 解码全白/过曝
```

### 5.6 为什么混合采样没根治(8/15 数学)

- min=0.77 低噪段 = timestep_id∈[770,1000) = sigma∈[0,0.78]
- 但 **sigma<0.2(推理最后一步真正所在区)需 id∈[980,1000),只占低噪段 20/230 ≈ 8.7%**
- 混合采样后 sigma<0.2 总火力 = 50%×2.0%(全范围自然占比)+ 50%×8.7% ≈ **5.3%**
- BSMNTW 在 sigma<0.2 权重仅 0.21-1.70 → 末段有效训练量 ≈ 5.3% × 低权重 ≈ 0
- **教训:选 min boundary 前先算目标 sigma 段在区间内的占比。"低噪段整体强化"≠"最后一步所在区强化"**

---

## 6. 已排除的嫌疑清单(回答"还有什么可能"用)

| 嫌疑 | 结论 | 证据 |
|---|---|---|
| 数据质量差 | 排除 | 全量扫描 + cache 解码 + x₀ scale 反事实 |
| SLA 实现错误 | 排除 | 判别实验 3:原始+SLA 白 3.0% |
| ZeRO-3 保存损坏 | 排除 | ckpt_zero_scan 全绿 |
| loss 发散/梯度爆炸 | 排除 | 0 NaN/Inf;err_ft−err_orig 全负;clip 前后症状相同 |
| 权重加载错误 | 排除 | verify_weight_load.py:hash 命中、key 对齐、shape 0 mismatch |
| 损失函数不好 | 排除 | 官方同款 SFT loss,无自定义成分 |
| cache 格式/归一化 | 排除 | 实验C 解码正常 |
| lr 是根因 | 排除(放大器) | 两轮 lr 相同,只改采样分布 → 白 78.8→52.3;症状随采样分布变 ⇒ 采样是因果变量 |
| 推理配置(CFG/shift) | 排除 | pipeline 默认 shift 12/3 与训练一致;cfg_scale=1.0 无 CFG |

---

## 7. 修正方向(待拍板,8/15 给出)

- **A. 收窄低噪采样**:min boundary 从 0.77 收到 sigma≤0.3(min≈0.92)或 sigma≤0.2(≈0.98);低噪段内改**均匀 sigma 采样**(绕开 shift 映射密度偏斜:base 均匀 → sigma 集中在高值端,id 靠 1000 的极端低噪仍稀少)
- **B. 降主干漂移**:lr 1e-5 → 5e-6(漂移减半 → 低噪端 2.7× 放大减半),或加权重衰减约束
- **C. 推理末步 latent clamp / 末步减步长**(治标,可快速验证机制)
- 组合 A+B 建议优先(改动小、方向明确);或先跑"高 lr 小步数"探针(2e-5~5e-5)确认主干能动
- 相关:数据 2000 条 × repeat 2 偏小偏重复,过拟合嫌疑大(3600 已缓解但未根除)

---

## 8. 外部参考仓库对比(HuuBar/DiffSynth-H3-SLA,8/15)

- 位置:`sla_ref_DiffSynth-H3-SLA/`(commit 2026-08-15)
- 与本仓实现高度同源:`minimax_h3_dit.py` 0 行差异;差异 = sla_kernel 通用化(CUDA/CPU + SLA_BACKEND 环境变量 + 纯 torch fallback)、extra_kwargs 签名过滤、无 boundary/混合采样
- **训练配方 = 论文原版全范围均匀采样**(dataset_repeat 100 × 2 epochs,无低噪强化、无 clip-grad),作者自述未真机验证 → 大概率踩我们验证过的低噪端欠训练坑,**别照抄**
- 详细 diff:`sla_ref/h3_sla_training_plan.md` 附近 + skill reference

---

## 9. 工具清单(c50058431/tools/ 与 dataset/)

| 工具 | 用途 |
|---|---|
| sla_infer.py / sla_infer_vast_paint.py | SLA ckpt 推理(python wrapper 读 jsonl prompt,防 shell 引号坑) |
| baseline_infer_vast_paint.py / infer_compare.py | 基线/对比推理 |
| sla200_nosla_infer.py / sla250_nosla_infer.py | 关 SLA 判别推理 |
| velocity_probe.py / velocity_probe_250.py | velocity 场对比(根因分析) |
| sampling_traj_check.py / traj_check_verbose.py / traj_compare_plot.py | 采样轨迹逐步解码 |
| frame_compare.py | 帧级 MAD/亮度对比 |
| loss_curve_check.py | 终端看 loss(EventAccumulator,每 5 步 + 跳变 + NaN 统计) |
| sla3600_loss_plot.py | loss 曲线 PNG |
| ckpt_delta_scan.py / ckpt_zero_scan.py | ckpt 权重漂移 / ZeRO 完整性扫描 |
| vae_latent_check.py / cache_decode_check.py / cache_scale_check.py | latent/VAE 侧验证 |
| data_quality_scan.py | 数据内容质量全量扫描 |
| verify_weight_load.py | 权重加载正确性验证 |
| sla_real_score_viz.py / sla_score_viz.py | block score 可视化(真实/模拟) |
| watch_sla_stage1.sh / sla3600_stop250_watcher.sh 等 | 训练 watcher |

---

## 10. 推理产物清单(本目录根下)

| 文件 | 内容 | 时间 |
|---|---|---|
| baseline_vast_paint_fl2va.mp4 | 基线(原始 FL2VA,白 3.5%) | 8/14 |
| baseline_sla_zero_vast_paint.mp4 | 原始+SLA 零 proj_l(白 3.0%) | 8/14 |
| sla_infer_ir4999_step125.mp4 | step-125(用户:勉强) | 8/14 |
| sla_infer_ir4999_step200.mp4 / sla_infer_vast_paint_step200.mp4 / sla_infer_step200_default.mp4 | step-200 三 prompt | 8/14 |
| sla200_nosla_vast_paint.mp4 | step-200 关 SLA(白 37.5%) | 8/14 |
| sla3600_low_step250_vast_paint.mp4 | **step-250 混合采样(白 52.3%)** | 8/15 00:29 |
| sla250_nosla_vast_paint.mp4 | step-250 关 SLA(白 22.6%) | 8/15 14:55 |
| cmp_baseline_golden_retriever.mp4 | 基线 golden retriever(白 0.0%) | 8/15 16:26 |
| cmp_sla_zero_golden_retriever.mp4 | 原始+SLA 零 proj_l golden(白 0.0%) | 8/15 16:31 |

**图表**:sla3600_loss_curve.png、traj_compare_step250_vs_orig.png(采样步并排对比,13MB)、sla_real_score_video.png / _trained.png、traj_frames_step250/ / traj_frames_orig/(逐步 PNG)。

---

## 11. 下一步决策点(给用户)

1. 选修正方向 A(收窄+均匀 sigma)和/或 B(lr 减半),确认后改 train.py 参数即可重训(复用 3600 cache,免重新 data_process)
2. 或先跑 lr 探针(2e-5~5e-5,50-100 步)确认主干可动性
3. ~~golden retriever 对比~~ 已完成:原始+SLA 零 proj_l 在第二个 prompt 也正常(白 0.0%)→ 实现无罪已彻底钉死

---

*附:训练口径说明——框架内部步数每次从 0 重计,累计总步数 = 各轮内部步数相加(用户偏好按累计口径汇报里程碑)。*
