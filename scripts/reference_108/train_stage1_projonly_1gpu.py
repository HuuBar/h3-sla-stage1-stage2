#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SLA stage1 单卡训练 (独立脚本): 冻结主干只训 proj_l, teacher 对齐, 200 步。

路线: 复用推理脚本已验证的 CPU offload 加载路径 (避开 train.py 单卡的 meta tensor 问题)。
- 加载: MiniMaxH3Pipeline.from_pretrained + ModelConfig(**cpu_vram), 单卡 npu
- 冻结: 主干 requires_grad=False, 仅 sla_module.proj_l 可训练 + SLA_PROJ_ONLY=1
- loss: TeacherAlignMiniMaxH3AudioVideoLoss (SLA_ALIGN_TEACHER=1, 每层 o_sla vs o_full MSE,
  梯度在 sla_core forward 内逐层 backward 累加)
- 保存: 只导出 proj_l 权重 (proj_l_stepN.pt), 主干零漂移 = 原始 FL2VA
"""
import glob, json, os, sys, time
import torch

if not torch.cuda.is_available() and hasattr(torch, "npu") and torch.npu.is_available():
    DEVICE = "npu"
    torch.npu.set_device(0)  # npu1: 由外层 ASCEND_RT_VISIBLE_DEVICES=1 映射
else:
    DEVICE = "cuda"
print(f"[stage1] device = {DEVICE}")

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.diffusion.loss import TeacherAlignMiniMaxH3AudioVideoLoss

os.environ["SLA_PROJ_ONLY"] = "1"
os.environ["SLA_ALIGN_TEACHER"] = "1"

BASE = "/mnt/a800_share/minimax-h3/FL2VA"
CACHE = "/mnt/a800_share/c50058431/dataset/ir5k_gen_latent_cache_480p_ckpt_200"
OUT = "/mnt/a800_share/c50058431/proj_l_align/proj_l_stage1_200"
LR = 1e-4
STEPS = 200

SLA_KW = dict(use_sla=True, sla_topk=0.05, sla_feature_map="softmax", sla_blkq=64, sla_blkk=64)

def _shards(subpath, pattern):
    return sorted(glob.glob(os.path.join(BASE, subpath, pattern)))

dtype = torch.bfloat16
cpu_vram = {
    "offload_dtype": dtype, "offload_device": "cpu",
    "onload_dtype": dtype, "onload_device": "cpu",
    "preparing_dtype": dtype, "preparing_device": DEVICE,
    "computation_dtype": dtype, "computation_device": DEVICE,
}
npu_total_gb = torch.npu.get_device_properties(0).total_memory / (1024**3)
vram_limit = npu_total_gb - 8
print(f"[stage1] vram_limit = {vram_limit:.1f} GiB")

print("[stage1] 加载模型 (CPU offload)...")
t0 = time.time()
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=dtype,
    device=DEVICE,
    model_configs=[
        ModelConfig(path=_shards("text_encoder", "model*.safetensors"), **cpu_vram),
        ModelConfig(path=_shards("transformer", "model*.safetensors"), extra_kwargs=SLA_KW, **cpu_vram),
        ModelConfig(path=_shards("video_vae/source", "model.safetensors"), **cpu_vram),
        ModelConfig(path=_shards("audio_vae", "model.safetensors"), **cpu_vram),
    ],
    processor_config=ModelConfig(path=os.path.join(BASE, "processor")),
    vram_limit=vram_limit,
)
print(f"[stage1] 模型加载完成 ({time.time()-t0:.0f}s). use_sla = {getattr(pipe.dit, 'use_sla', False)}")

# 冻结主干, 只留 proj_l
n_proj = 0
n_params = 0
for name, p in pipe.dit.named_parameters():
    keep = "sla_module.proj_l" in name
    p.requires_grad_(keep)
    if keep:
        n_proj += 1
        n_params += p.numel()
print(f"[stage1] 冻结主干: {n_proj} 个 proj_l 参数对象, {n_params} params 可训练")

# proj_l 是 VRAM offload 包装的 meta 占位 (原始 FL2VA 无 proj_l 权重), 显式零初始化 materialize
# VRAMLinear.cast_to 对 meta weight 会崩 (Cannot copy out of meta tensor), 直接填数据
n_fixed = 0
for blk in pipe.dit.blocks:
    sla = getattr(blk.attn, "sla_module", None)
    if sla is not None and hasattr(sla, "proj_l"):
        pl = sla.proj_l
        w = getattr(pl, "weight", None)
        if w is not None and w.is_meta:
            pl.weight = torch.nn.Parameter(torch.zeros(w.shape, dtype=torch.float32))
            n_fixed += 1
        b = getattr(pl, "bias", None)
        if b is not None and b.is_meta:
            pl.bias = torch.nn.Parameter(torch.zeros(b.shape, dtype=torch.float32))
print(f"[stage1] proj_l meta materialize: {n_fixed} 个 weight")

# 数据
files = sorted(glob.glob(os.path.join(CACHE, "*.pth")))
print(f"[stage1] 数据: {len(files)} 条 (200 步 = 每步 1 条)")

# optimizer (只含 proj_l 参数)
proj_params = [p for name, p in pipe.dit.named_parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(proj_params, lr=LR, weight_decay=1e-2)

# 训练循环
pipe.train()
scheduler = pipe.scheduler
scheduler_audio = pipe.scheduler_audio
# teacher align 训练分布 = 推理分布: 50 点网格 (与 train.py teacher_align 模式一致)
scheduler.set_timesteps(50, training=True, shift=12.0)
scheduler_audio.set_timesteps(50, training=True, shift=3.0)
print(f"[stage1] 50 点推理网格: sigma[0]={scheduler.sigmas[0]:.3f} sigma[-1]={scheduler.sigmas[-1]:.4f}")

losses = []
for step in range(STEPS):
    f = files[step % len(files)]
    d = torch.load(f, map_location="cpu", weights_only=False)
    shared, embeds, _ = d
    inputs = dict(shared)
    inputs.update(embeds)
    inputs["use_gradient_checkpointing"] = False  # teacher align 必须关
    inputs["use_gradient_checkpointing_offload"] = False
    inputs["max_timestep_boundary"] = 1.0
    inputs["min_timestep_boundary"] = 0.0
    # 转 device + dtype (与 train.py transfer_data_to_device 一致, 递归处理嵌套 dict)
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

    optimizer.zero_grad()
    loss = TeacherAlignMiniMaxH3AudioVideoLoss(pipe, **inputs)
    # 梯度已在各层 sla forward 内 backward 累加; loss 是 detach 日志值
    losses.append(float(loss))
    optimizer.step()
    if torch.npu.is_available():
        torch.npu.empty_cache()

    if (step + 1) % 10 == 0:
        avg = sum(losses[-10:]) / min(10, len(losses))
        print(f"[stage1] step {step+1}/{STEPS} loss(近10均)={avg:.4f} ({time.time()-t0:.0f}s)")

# 保存 proj_l (只导出 proj_l, 主干零漂移 = 原始 FL2VA)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
state = {name: p.detach().cpu() for name, p in pipe.dit.named_parameters() if "sla_module.proj_l" in name}
torch.save(state, OUT + ".pt")
print(f"[stage1] 已保存 proj_l: {OUT}.pt ({len(state)} keys)")
mean_w = torch.cat([state[k].flatten().abs() for k in state if "weight" in k]).mean().item()
print(f"[stage1] proj_l mean|w| = {mean_w:.6f}")
