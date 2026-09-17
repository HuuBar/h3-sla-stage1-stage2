#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""proj_l 专用对齐训练:
- 主干(原始 FL2VA)全冻结, 只训 proj_l (0.83M 参数)
- loss = 每层 attention 输出对齐: MSE(o_sla, o_full), teacher = 全注意力 (同 q/k/v)
- timestep 从 50 步推理网格均匀采样 (训练分布 = 推理分布)
- 单卡 CPU offload, 数据 = 3600 split-cache
"""
import argparse, glob, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["SLA_PROJ_ONLY"] = "1"  # sla_core: o_s detach, 省 sparse backward (数学等价, 主干冻结)

if not torch.cuda.is_available() and hasattr(torch, "npu") and torch.npu.is_available():
    DEVICE = "npu"
    torch.npu.set_device(0)
else:
    DEVICE = "cuda"
print(f"[proj] device = {DEVICE}")

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

MODEL_BASE = "/mnt/a800_share/minimax-h3/FL2VA"
CACHE = "/mnt/a800_share/c50058431/minimax_dmd/models/train/MiniMax-H3-T2VA-SLA-3600-split-cache"

def _shards(subpath, pattern):
    return sorted(glob.glob(os.path.join(MODEL_BASE, subpath, pattern)))

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--out", default="/mnt/a800_share/c50058431/proj_l_align")
ap.add_argument("--save-every", type=int, default=100)
ap.add_argument("--init-proj", default=None, help="加载已有 proj_l state dict 作为初始权重 (续跑)")
ap.add_argument("--start-step", type=int, default=0, help="总步数偏移 (续跑时用于保存文件名)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

# ---------- 1. 构造 pipeline: 原始 FL2VA + SLA ----------
dtype = torch.bfloat16
cpu_vram = {
    "offload_dtype": dtype, "offload_device": "cpu",
    "onload_dtype": dtype, "onload_device": "cpu",
    "preparing_dtype": dtype, "preparing_device": DEVICE,
    "computation_dtype": dtype, "computation_device": DEVICE,
}
# vram_limit 调小: check_free_vram() 在少量层上卡后返回 False -> 层参数流式临时 cast
# (与推理一致, 用完释放), 避免梯度训练时 50 层参数全部 preparing 常驻 NPU 累积 59GB
vram_limit = 5.0
print(f"[proj] vram_limit = {vram_limit:.1f} GiB (流式 offload 模式)")

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=dtype,
    device=DEVICE,
    model_configs=[
        ModelConfig(path=_shards("text_encoder", "model*.safetensors"), **cpu_vram),
        ModelConfig(path=_shards("transformer", "model*.safetensors"),
                    extra_kwargs=dict(use_sla=True, sla_topk=0.05, sla_feature_map="softmax",
                                      sla_blkq=64, sla_blkk=64), **cpu_vram),
        ModelConfig(path=_shards("video_vae/source", "model.safetensors"), **cpu_vram),
        ModelConfig(path=_shards("audio_vae", "model.safetensors"), **cpu_vram),
    ],
    processor_config=ModelConfig(path=os.path.join(MODEL_BASE, "processor")),
    vram_limit=vram_limit,
)
assert getattr(pipe.dit, "use_sla", False), "SLA 未启用!"

# ---------- 2. proj_l 零初始化 (原始权重无 proj_l key -> meta 占位) ----------
with torch.no_grad():
    for blk in pipe.dit.blocks:
        m = blk.attn.sla_module.proj_l
        m.weight = nn.Parameter(torch.zeros(128, 128, dtype=torch.float32))
        m.bias = nn.Parameter(torch.zeros(128, dtype=torch.float32))
print("[proj] proj_l 零初始化完成")

# 续跑: 加载已有 proj_l 权重
if args.init_proj:
    sd = torch.load(args.init_proj, map_location="cpu")
    loaded = 0
    with torch.no_grad():
        for i, blk in enumerate(pipe.dit.blocks):
            for pn, p in blk.attn.sla_module.proj_l.named_parameters():
                key = f"blocks.{i}.attn.sla_module.proj_l.{pn}"
                if key in sd:
                    p.data.copy_(sd[key])
                    loaded += 1
    print(f"[proj] 加载初始 proj_l: {args.init_proj} ({loaded} keys)")

# ---------- 3. 冻结主干, 只开 proj_l ----------
proj_params = []
for blk in pipe.dit.blocks:
    for p in blk.attn.sla_module.proj_l.parameters():
        proj_params.append(p)
for p in pipe.dit.parameters():
    p.requires_grad_(False)
for p in proj_params:
    p.requires_grad_(True)
n_trainable = sum(p.numel() for p in pipe.dit.parameters() if p.requires_grad)
print(f"[proj] 可训练参数 = {n_trainable/1e6:.2f}M (应 ≈0.83M)")

# ---------- 4. hooks: 每层 sla_module 的 pre 算 o_full (teacher), post 收集 o_sla ----------
align_pairs = []
SCALE = 128 ** -0.5
CHUNK = 512  # 全注意力分块: 避免 QK^T 物化 (15185^2*56*2B ≈ 26GB) 爆显存

def _full_attn(q, k, v, scale, chunk=CHUNK):
    """分块全注意力 (与 torch_sdpa 等价的逐块 SDPA)"""
    B, H, L, D = q.shape
    outs = []
    for s in range(0, L, chunk):
        outs.append(F.scaled_dot_product_attention(q[:, :, s:s + chunk], k, v, scale=scale))
    return torch.cat(outs, dim=2)

def _make_hooks(blk):
    buf = {}
    proj = blk.attn.sla_module.proj_l

    def pre(mod, args):
        seg_q, seg_k, seg_v = args  # (1,H,L,D) bf16
        with torch.no_grad():
            buf["o_full"] = _full_attn(seg_q, seg_k, seg_v, SCALE)

    def post(mod, args, output):
        o_sla = output[0] if isinstance(output, tuple) else output
        o_l = getattr(mod, "_o_l", None)
        if "o_full" in buf and o_l is not None:
            with torch.no_grad():
                o_s = o_sla - proj(o_l.to(torch.float32)).to(o_sla.dtype)  # 反推纯稀疏输出
            align_pairs.append((o_s.cpu(), o_l.cpu(), buf.pop("o_full").cpu(), proj))

    blk.attn.sla_module.register_forward_pre_hook(pre)
    blk.attn.sla_module.register_forward_hook(post)

for blk in pipe.dit.blocks:
    _make_hooks(blk)
print(f"[proj] hooks 注册完成 ({len(pipe.dit.blocks)} blocks)")

# ---------- 5. 数据 ----------
files = sorted(glob.glob(os.path.join(CACHE, "*", "*.pth")))
random.shuffle(files)
print(f"[proj] cache 样本 = {len(files)}")

# ---------- 6. 训练 ----------
opt = torch.optim.AdamW(proj_params, lr=args.lr, weight_decay=0.0)
sched = pipe.scheduler
sched.set_timesteps(50, training=False)  # 50 步推理网格
sched_a = pipe.scheduler_audio
sched_a.set_timesteps(50, training=False)
sig_v, sig_a = sched.sigmas, sched_a.sigmas
print(f"[proj] 50 点网格: sigma[0]={sig_v[0]:.3f} sigma[49]={sig_v[49]:.4f}")

models = {"dit": pipe.dit}
t_start = time.time()
log_f = os.path.join(args.out, "train.log")

def _mem(tag):
    a = torch.npu.memory_allocated() / 1e9
    r = torch.npu.memory_reserved() / 1e9
    print(f"[mem] {tag}: allocated={a:.2f}G reserved={r:.2f}G", flush=True)

_mem("构造后")

for step in range(1, args.steps + 1):
    f = files[(step - 1) % len(files)]
    shared, posi, _ = torch.load(f, map_location="cpu")
    input_latents = shared["input_latents"].to(device=DEVICE, dtype=dtype)
    audio_input = shared["audio_input_latents"].to(device=DEVICE, dtype=dtype)
    prompt_embeds = posi["prompt_embeds"].to(device=DEVICE, dtype=dtype)
    packed = {k: (v.to(device=DEVICE) if torch.is_tensor(v) else v)
              for k, v in posi["packed"].items()}

    idx = random.randrange(50)
    tv = sched.timesteps[idx].cpu()
    ta = sched_a.timesteps[idx].cpu()
    t_video = float(1.0 - sig_v[idx])
    t_audio = float(1.0 - sig_a[idx])

    noise = torch.randn_like(input_latents)
    latents = sched.add_noise(input_latents, noise, tv)
    audio_noise = torch.randn_like(audio_input)
    audio_latents = sched_a.add_noise(audio_input, audio_noise, ta)

    align_pairs.clear()
    if step == 1:
        _mem("forward前")
    # 主干 forward 全程 no_grad (显存 = 推理级); proj_l 分支下面梯度模式重算
    with torch.no_grad():
        noise_pred, noise_pred_audio = pipe.model_fn(
            **models,
            video_latents=latents,
            audio_latents=audio_latents,
            packed=packed,
            prompt_embeds=prompt_embeds,
            t_video=t_video, t_audio=t_audio,
            imgvid_cond_noise_aug=shared["imgvid_cond_noise_aug"],
            audio_cond_noise_aug=shared["audio_cond_noise_aug"],
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
            device=DEVICE,
        )
    # 梯度模式: 重算 proj_l 分支并对齐 teacher, 逐层 backward (图用完即释放, 避免 50 层全挂图 OOM)
    torch.npu.empty_cache()
    if step == 1:
        _mem("forward后")
    opt.zero_grad(set_to_none=True)
    loss_sum = 0.0
    n_pairs = 0
    for o_s, o_l, o_full, proj in align_pairs:
        o_s = o_s.to(DEVICE)
        o_l = o_l.to(DEVICE)
        o_full = o_full.to(DEVICE)
        o_sla = o_s + proj(o_l.to(torch.float32)).to(o_s.dtype)
        loss_i = F.mse_loss(o_sla.float(), o_full.float())
        loss_i.backward()  # 每层独立图, 只更新该层 proj_l 梯度
        loss_sum += loss_i.item()
        n_pairs += 1
        del o_s, o_l, o_full, o_sla, loss_i
    opt.step()
    loss = loss_sum / max(n_pairs, 1)

    if step % 10 == 0:
        torch.npu.empty_cache()

    if step % 10 == 0 or step == 1:
        mw = sum(p.abs().mean().item() for p in proj_params) / len(proj_params)
        msg = (f"[proj] step {step} loss {loss:.6f} proj|w| {mw:.4f} "
               f"pairs {len(align_pairs)} elapsed {time.time()-t_start:.0f}s")
        print(msg, flush=True)
        with open(log_f, "a") as lf:
            lf.write(msg + "\n")

    if step % args.save_every == 0:
        sd = {}
        for i, blk in enumerate(pipe.dit.blocks):
            for pn, p in blk.attn.sla_module.proj_l.named_parameters():
                sd[f"blocks.{i}.attn.sla_module.proj_l.{pn}"] = p.detach().cpu()
        pth = os.path.join(args.out, f"proj_l_step{args.start_step + step}.pt")
        torch.save(sd, pth)
        print(f"[proj] saved {pth}", flush=True)

print(f"[proj] done, total {time.time()-t_start:.0f}s")
