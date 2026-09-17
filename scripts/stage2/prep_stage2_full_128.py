#!/usr/bin/env python3
"""128 机: 从全量 768p latent 缓存构建 stage2 训练缓存 (全量版, 不过滤旧样本) (2026-09-01).

- 遍历 ir5k_gen_latent_cache_768p 全部 .pth (约 4959 条), 损坏的跳过;
- 校验 shape (1,24,37,48,84), 置 use_gradient_checkpointing=True, 输出 stage2_full。

用法 (h3_train_hang 容器内):
  python3 pkg_128_infer/scripts/prep_stage2_full_128.py [SRC] [DST]
"""
import glob, os, sys, torch

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/share/c50058431/c50058431/dataset/ir5k_gen_latent_cache_768p"
DST = sys.argv[2] if len(sys.argv) > 2 else "/mnt/share/c50058431/c50058431/dataset/ir5k_gen_latent_cache_768p_stage2_full"

os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC, "*.pth")))
print(f"[prep-full] 源文件数 {len(files)}", flush=True)

bad_shape, done, skipped = [], 0, 0
for i, f in enumerate(files):
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [SKIP] {f}: 损坏/未写完 ({e})", flush=True)
        skipped += 1
        continue
    if not isinstance(d, (list, tuple)) or len(d) < 2:
        print(f"  [SKIP] {f}: 结构异常", flush=True)
        skipped += 1
        continue
    shared, embeds, _ = d[0], d[1], (d[2] if len(d) > 2 else {})
    shape = tuple(shared["input_latents"].shape)
    if shape != (1, 24, 37, 48, 84):
        bad_shape.append((os.path.basename(f), shape))
        skipped += 1
        continue
    shared["use_gradient_checkpointing"] = True
    torch.save((shared, embeds, {}), os.path.join(DST, os.path.basename(f)))
    done += 1
    if i % 300 == 0:
        print(f"  [{i}] done={done} skipped={skipped}", flush=True)

print(f"[prep-full] 完成 {done} 个 -> {DST}, 跳过 {skipped} 个", flush=True)
if bad_shape:
    print(f"[prep-full] 非768p shape 的 {len(bad_shape)} 个 (已跳过): {bad_shape[:10]}", flush=True)
print(f"[prep-full] 最终可用条数 = {done} ({done}/16 = {done/16:.1f} 步/遍)", flush=True)
