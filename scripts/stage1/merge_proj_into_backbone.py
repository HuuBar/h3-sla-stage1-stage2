#!/usr/bin/env python3
"""把 proj-only 训练产物(step-N.safetensors, 100 proj_l keys)合并回完整 ckpt。
用法: python3 merge_proj_into_backbone.py <full_ckpt> <proj_ckpt> <out_ckpt>
"""
import sys, time
import torch
from safetensors.torch import load_file, save_file

full, proj, out = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
torch.set_num_threads(128)
sd = load_file(full)
p = load_file(proj)
n = 0
for k, v in p.items():
    if k in sd:
        sd[k] = v.to(sd[k].dtype)
        n += 1
    else:
        print(f"[merge] 跳过(主干无此 key): {k}", flush=True)
print(f"[merge] merged {n}/{len(p)} keys into {len(sd)}-key ckpt", flush=True)
save_file(sd, out)
print(f"[merge] done: {out} in {time.time()-t0:.0f}s", flush=True)
