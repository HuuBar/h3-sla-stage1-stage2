""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import torch
import triton
import triton.language as tl


"""
func tmp_compress_kernel

do the same thing as `compress_kernel`: compress the L dimension of X, try
to solve "too many grid num"

Max L is 524288, so idx_l can be 8191, this would cause too many grids
running simutanously. To avoid this, change it to a constant irrelavent
to idx_l.

reference:
    triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html

params:
    _program_id: in range (0, 0) to (B * H, nproc)
    x: in shape [B][H][L][D]
    mean_x: in shape [B][H][(L+BLOCK_L-1)//L_BLOCKS][D]
"""
@triton.jit
def compress_kernel(
    x, mean_x,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr
):
    idx_bh = tl.program_id(0)
    nproc = tl.program_id(1)

    # the third dimension of XM
    comp_l_len = (L + BLOCK_L - 1) // BLOCK_L
    # allocate `nproc` evenly to handle `comp_l_len` blocks,
    # where comp for compressed
    # for example, nproc = 32, current proc = 3:
    # handle comp_l: 3, 32 + 3, 32 * 2 + 3, ...
    comp_l_start = nproc
    comp_l_step  = tl.num_programs(1)
    for comp_l_idx in tl.range(
        comp_l_start, comp_l_len, comp_l_step,
        # num_stages=2
    ):
        l_idx        = comp_l_idx * BLOCK_L # index of x dimension 3
        start_x      = x + idx_bh * L * D + l_idx * D
        start_mean_x = mean_x + idx_bh * comp_l_len * D + comp_l_idx * D

        # load x range: [][][BLOCK_L][D]
        range_x = tl.arange(0, BLOCK_L)[:, None] * D + tl.arange(0, D)[None, :]
        mask_x  = l_idx + tl.arange(0, BLOCK_L)[:, None] < L
        # save mean_x range: [][][1][D]
        range_mean_x = tl.arange(0, D)

        # load, compute, save
        i      = tl.load(start_x + range_x, mask=mask_x) # shape: (BLOCK_L, D)
        len_i  = min(BLOCK_L, L - l_idx)
        mean_i = tl.sum(i, axis=0) / len_i
        tl.store(start_mean_x + range_mean_x, mean_i)


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)
    # 910B1: CUBE 24, VECTOR 48

    grid = (B * H, min(L_BLOCKS, 32768 // (B * H)))
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64, force_critical_mask=None):
    arg_k = k - torch.mean(k, dim=-2, keepdim=True) # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    # H3 短段保护: 块数少时 topk_ratio 取整可能为 0 -> 无块可选 -> nan. 至少选 1 块.
    topk = max(1, min(K, int(topk_ratio * K)))
    if force_critical_mask is not None:
        # 模态保护(列方向): 文本/音频 key 块对所有 query 行强制可见.
        # 把 force 列的 score 置 +inf 并扩大 topk -> 这些列必被选中, 且 lut 保持定长, kernel 无需改动.
        # 每行 force 列集合相同(列级保护), 取列维 OR 得到统一列集.
        force_critical_mask = force_critical_mask.to(device=pooled_score.device, dtype=torch.bool)
        # (B,H,NQ,NK) -> 列维 OR(任意 query 行保护该列) -> head 维 OR(所有 head 保护列相同)
        force_cols = force_critical_mask.any(dim=-2).any(dim=1, keepdim=True)  # (B,1,NK)
        n_force = int(force_cols.sum().item())
        new_topk = min(K, topk + n_force)
        masked_score = pooled_score.masked_fill(
            force_cols.unsqueeze(-2).expand_as(pooled_score), float("inf"))
        lut = torch.topk(masked_score, new_topk, dim=-1, sorted=False).indices
        sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
        sparse_map.scatter_(-1, lut, 1)
        return sparse_map, lut, new_topk
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk







