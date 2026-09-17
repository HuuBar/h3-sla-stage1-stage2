#!/usr/bin/env python
"""Merge DMD2 step-750 LoRA into SLA step-156 transformer (single-file ckpt).

W' = W_sla + lora_B @ lora_A  (fp32 delta, cast back to W dtype)
LoRA: /mnt/share/h00944024/minimax_dmd/outputs/step-750.safetensors
SLA : /mnt/share/c50058431/c50058431/minimax_dmd/models/train/MiniMax-H3-T2VA-SLA-768p-2493-156/step-156.safetensors
OUT : same dir / step-156-sla-dmd.safetensors
"""
import os, sys, time
import torch
from safetensors.torch import load_file, save_file

LORA_PATH = "/mnt/share/h00944024/minimax_dmd/outputs/step-750.safetensors"
SLA_PATH = "/mnt/share/c50058431/c50058431/minimax_dmd/models/train/MiniMax-H3-T2VA-SLA-768p-2493-156/step-156.safetensors"
OUT_PATH = "/mnt/share/c50058431/c50058431/minimax_dmd/models/train/MiniMax-H3-T2VA-SLA-768p-2493-156/step-156-sla-dmd.safetensors"

def main():
    t0 = time.time()
    torch.set_num_threads(128)
    print("[1/3] loading LoRA ...", flush=True)
    lora = load_file(LORA_PATH)
    lora_pairs = {}
    for k in lora:
        if ".lora_A.default.weight" in k:
            bk = k.replace(".lora_A.default.weight", ".weight")
            bk_b = k.replace("lora_A.default", "lora_B.default")
            lora_pairs[bk] = (lora[k], lora[bk_b])
    print(f"      {len(lora_pairs)} LoRA targets", flush=True)

    print("[2/3] loading SLA step-156 (66GB) ...", flush=True)
    sd = load_file(SLA_PATH)
    print(f"      {len(sd)} keys, {time.time()-t0:.0f}s", flush=True)

    merged = 0
    for bk, (A, B) in lora_pairs.items():
        if bk in sd:
            W = sd[bk]
            delta = (B.float() @ A.float()).to(W.dtype)
            sd[bk] = W + delta
            merged += 1
    print(f"[3/3] merged {merged}/{len(lora_pairs)} targets, saving ...", flush=True)
    save_file(sd, OUT_PATH)
    print(f"done: {OUT_PATH} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    sys.exit(main())
