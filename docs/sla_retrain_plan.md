# H3 SLA 微调重启方案 (2026-08-14, 白屏根因收官后)

## 1. 根因结论 (四步交叉验证, 详见 tools/ 与 skill)

- 数据侧排除: 2000 条全扫, 白% 中位 2.6%, 数据不白 (低纹理 36.5% 是"模糊"助推剂)
- 权重侧排除: step-200 vs FL2VA 无 NaN/Inf, 主干 relΔ ~0.1%
- latent 实锤: 白屏 = latent 正向漂移 + 方差坍缩 (|z| 0.38→0.55→0.75, std 0.84→0.82→0.68)
- loss 旁证: batch=1 逐步抖动正常, 但第二轮 loss 均值 0.442→0.464→0.528 反弹 = 过拟合区

机制: 33B 全参 lr1e-5 × 200 步剂量不足 (主干只动 0.1%) + 2000 条 × repeat2 过拟合
+ 无两阶段 timestep boundary → 学出来的微调把模型推向采样不稳定区, 50 步迭代累积成
latent 漂移 → 解码全白。SLA 推理侧是放大器 (37.5% → 79.4%), 不是根因。

## 2. 修复配方 (对齐 MindSpeed 已验证配置, sla_ref/h3_sla_training_plan.md §2)

| 项 | MindSpeed 已验证 (Wan A14B) | 我们上次 | 本次方案 |
|---|---|---|---|
| 阶段1 (高噪) | max_timestep_boundary=0.417, min=0, 500 it | 无 (默认 1/0 全范围) | 0.417/0, 500 it |
| 阶段2 (低噪) | max=1.0, min=0.417, 500 it | 无 | 1.0/0.417, 500 it |
| lr | 1e-5 恒定, wd 1e-2, clip 1.0 | 1e-5 | 1e-5 (里程碑若主干仍不动 → 2e-5) |
| 数据 | 5000 条/阶段 | 2000 条 × repeat 2 | ir5k 全量 ~4724, repeat 2 (≈590 it, 停 500) |
| 全参微调 | 是 | 是 | 是 |
| SLA 配置 | topk 0.05, softmax, blk 64/64 | 同 | 同 (proj_l 保持 kaiming) |
| 推理 | 走 SLA | 走 SLA (方向正确) | 走 SLA, 里程碑自动推理 |

## 3. 执行步骤

### 3.1 数据: h3_train_data_ir5k_4724 (新增目录, ~15-20 min)
- 复制 dataset/build_ir5k_subset_2000.py → 参数化 (TARGET=4724, OUT_ROOT=..._4724)
- 源: dataset/clips_ir5k (4724 已下载) + dataset/dl_list_ir5k.jsonl
- 可选: 从 quality_scan 剔除 STATIC(2.9%)/全黑/全白/静音样本

### 3.2 代码改动 (最小化, 需批准)
1. train.py: parser 加 `--max-timestep-boundary` / `--min-timestep-boundary`
   (float, 默认 1.0/0.0), get_pipeline_inputs 注入 inputs_shared
   (loss.py:66-94 已支持, 无需改 loss)
2. 新启动脚本 SLA-high.sh / SLA-low.sh (或给现有 .sh 加 stage 参数):
   - high: boundary 0.417/0, dataset_repeat 2, OUT=...-SLA-high, save_steps 50
   - low: boundary 1.0/0.417, 从 high 的 step-500 启动 (model_paths 法, 禁 --resume)
   - 全部 dataset_repeat 2 → 590 it/rank, 用 stop-at-500 watcher 精确停

### 3.3 high→low 交接 (零3下 --resume 必崩, 用已验证的 hash 注册 + model_paths 法)
1. model_configs.py 注册 SLA ckpt 的 hash (922fd0b8..., 一次注册永久可用)
2. mkdir ...-SLA-low/transformer && ln -sf <high>/step-500.safetensors
3. low 用 --model_paths "[glob(该目录)]" 启动, 不带 --resume

### 3.4 smoke test (启动后先验 ~50-100 步, ~1.5-2.5h)
- loss 趋势 (EventAccumulator 分窗均值, 不要看单步抖动), sla_sparsity ~0.9
- 无 crash (OOM/MTE OOB/HCCL), 再放全量

### 3.5 里程碑自动评估 (watcher, 复用 tools/sla_pipeline_watcher.sh 模式)
- 250 / 500 (stage1 末) / 750 / 1000 (stage2 末): 自动 sla_infer (python wrapper
  读 jsonl prompt) + 白% 测量 (tools/data_quality_scan.py 单文件模式或 cv2 直测)
- 500 时检查: 主干 delta (ckpt_delta_scan.py) + 白% —— 若主干仍 ~0.1% 且白,
  阶段2 提 lr 到 2e-5

### 3.6 节奏
- cache 生成 (stage1 data_process, 8 ranks, ~4724 条 ≈ 2.5h)
- 阶段1 500 it × 83s/it ≈ 11.5h (16 ranks)
- 阶段2 500 it ≈ 11.5h
- 总 ≈ 26h + 中间评估。可分天: 今晚 cache + stage1, 明早看 500 评估再定 stage2 lr

## 4. 风险与注意

- 续训禁 --resume_from_checkpoint (zero3 size mismatch 崩), 一律 model_paths 法
- save_steps 覆盖 hazard: high/low 用不同 OUT 目录
- 62 GB/ckpt: save_steps 50 → 10 ckpt/stage ≈ 620 GB/stage, 预留磁盘
- 停步用 pkill watcher (kill 后 OUT 只留最后 saved step, 无半截 ckpt)
- proj_l 保持 kaiming, 不改回论文零初始化 (零初始化时 q/k/v 梯度数学上 100%
  只走 sparse 路径, 是当初梯度饿死的根源)
- 用户口径: 里程碑按累计总步数汇报 (high 500 = 总 500, low 内 250 = 总 750)

## 5. 诊断收官: 采样轨迹实锤 (2026-08-14 晚)

tools/sampling_traj_check.py (monkeypatch pipe.step 逐步记录视频 latent + 独立 VAE 解码):

- step 1-45: step-200 与原始 FL2VA 的 |z|/std 几乎一致(差 <0.03) —— 0.1% 权重变化
  在轨迹前 90% 无感
- step 49 (最后一步, t→0 低噪端): step-200 |z| 0.62→0.99 爆炸、范围 ±4.75 (前 45 步
  都在 ±3.33 内), 解码亮 0.877 / 白 69.2%; 原始只到 0.633 / 8.8%
- 结论: 白屏 = 训练把低噪端 velocity 场学偏 (幅度过大) → 最后一步 overshoot →
  latent 冲出正常范围 → 解码全白。一步之差, 不是全程漂移。
- 直接验证两阶段 boundary 配方: low 阶段 (min=0.417) 就是专训低噪端的。
- ZeRO-3 保存完整性已排除 (ckpt_zero_scan.py 全绿; blocks.49.adaln_proj 的 22% 零是
  原始结构, 零位置与 FL2VA 100% 一致)

⚠️ 轨迹脚本两个坑: ① 只记录 `scheduler is pipe.scheduler` 的视频步 (音频 [2,32,207]
会覆盖 5D 视频张量); ② 解码用独立 VAE 实例 (pipe.video_vae 受 vram offload 影响
decode_temporal 返回 None)。
