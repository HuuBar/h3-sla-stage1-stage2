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
import torch_npu
import triton.runtime.driver as driver
device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]


@triton.jit
def _attn_bwd_preprocess(
    o_s, do_s, delta_s,
    BHL: tl.constexpr,
    D: tl.constexpr,
):
    """preprocess of attention backward

    Args:
        grid (Tuple[int]): nproc grids

        o_s (Tensor(B, H, L, D)): ptr to data 1
        do_s (Tensor(B, H, L, D)): ptr to data 2
        delta_s (Tensor(B, H, L)): ptr to target

    TODO:
        - is o_s and do_s contigous
    """
    pid   = tl.program_id(0)
    nproc = tl.num_programs(0)
    # TODO: tuned for 192K UB
    BLOCK_M: tl.constexpr = 128
    # treat o_s, delta_s as (B * H * L, D)
    idx_start, idx_step = pid * BLOCK_M, nproc * BLOCK_M
    for idx in tl.range(
        idx_start, BHL, idx_step
    ):
        range_input  = (
            idx * D +
            tl.arange(0, BLOCK_M)[:, None] * D + # row
            tl.arange(0, D)[None, :] # col
        )
        mask_input   = (idx + tl.arange(0, BLOCK_M))[:, None] < BHL
        range_output = idx + tl.arange(0, BLOCK_M)
        mask_output  = (idx + tl.arange(0, BLOCK_M)) < BHL

        in_1 = tl.load(o_s  + range_input, mask=mask_input)
        in_2 = tl.load(do_s + range_input, mask=mask_input)
        out  = tl.sum(in_1 * in_2, axis=1).to(delta_s.type.element_ty)

        tl.store(delta_s + range_output, out, mask=mask_output)


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    Q, K, V, LSE, DELTAS,
    DOS, DQ, LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sub_BN: tl.constexpr,   # BLOCK_N//2
):
    NUM_BLOCKS_M = M_BLOCKS
    NUM_BLOCKS = M_BLOCKS * B * H
    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, 20):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        qkv_offset = task_hz_idx.to(tl.int64) * L * D
        lse_offset = task_hz_idx.to(tl.int64) * L
        lut_offset = (task_hz_idx.to(tl.int64) * NUM_BLOCKS_M + task_m_idx.to(tl.int64)) * topk

        K_base = K + qkv_offset
        V_base = V + qkv_offset
        Q_base = Q + qkv_offset
        DOS_base = DOS + qkv_offset
        DQ_base = DQ + qkv_offset

        LSE_base = LSE + lse_offset
        DELTA_base = DELTAS + lse_offset
        LUT_ptr = LUT + lut_offset

        # ptrs for full BM
        Q_ptrs = Q_base + offs_m[:, None] * D + offs_d[None, :]
        DOS_ptrs = DOS_base + offs_m[:, None] * D + offs_d[None, :]
        DQ_ptrs = DQ_base + offs_m[:, None] * D + offs_d[None, :]

        LSE_ptrs = LSE_base + offs_m
        DELTA_ptrs = DELTA_base + offs_m

        # BM一次加载
        q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)                     # bf16 [BM, D]
        do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L)                # bf16 [BM, D]
        lse = tl.load(LSE_ptrs, mask=offs_m < L, other=float("inf"))      # fp32 [BM]
        delta_s = tl.load(DELTA_ptrs, mask=offs_m < L)                    # fp32 [BM]

        dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)                     # fp32 [BM, D]

        for block_idx_topk in tl.range(topk, num_stages=2):
            idx_n = tl.load(LUT_ptr + block_idx_topk)  # scalar

            # sub_block
            for n_part in tl.static_range(0, BLOCK_N, sub_BN):
                offs_n = n_part + tl.arange(0, sub_BN)
                n_mask = offs_n < (L - idx_n * BLOCK_N)

                K_ptrs2 = K_base + (idx_n * BLOCK_N + offs_n)[:, None] * D + offs_d[None, :]
                V_ptrs2 = V_base + (idx_n * BLOCK_N + offs_n)[:, None] * D + offs_d[None, :]

                k = tl.load(K_ptrs2, mask=n_mask[:, None])  # bf16 [sub_BN, D]
                v = tl.load(V_ptrs2, mask=n_mask[:, None])  # bf16 [sub_BN, D]

                qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)     # [BM, sub_BN] fp32
                p = tl.math.exp2(qk - lse[:, None])
                # qk = tl.dot(q, k.T) * qk_scale     # [BM, sub_BN] fp32
                # p = tl.math.exp(qk - lse[:, None])

                p = tl.where(n_mask[None, :], p, 0.0)

                dp = tl.dot(do_s, v.T).to(tl.float32)                      # [BM, sub_BN] fp32
                ds = p * (dp - delta_s[:, None])                           # [BM, sub_BN] fp32

                dq += tl.dot(ds.to(k.dtype), k)

        tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L)

@triton.jit
def _attn_bwd_dkdv_inner(
        Q_ptrs, k, v, DOS_ptrs, LSE_ptrs,
        DELTAS_ptrs, KBID_ptrs, DK_ptrs, DV_ptrs,
        qk_scale,
        L: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        N_BLOCKS: tl.constexpr,
        ):

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    for idx_m in tl.range(0, L, BLOCK_M):
        kbid = tl.load(KBID_ptrs)

        if kbid == 1:
            q = tl.load(Q_ptrs, boundary_check=(0,))
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)
            lse = tl.load(LSE_ptrs, boundary_check=(0,))
            pT = tl.math.exp2(qkT - lse[None, :])

            do = tl.load(DOS_ptrs, boundary_check=(0,))
            dv += tl.dot(pT.to(do.dtype), do) + 1e-14

            dpT = tl.dot(v, tl.trans(do))
            delta = tl.load(DELTAS_ptrs, boundary_check=(0,))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q) + 1e-14

        Q_ptrs = tl.advance(Q_ptrs, (BLOCK_M, 0))
        DOS_ptrs = tl.advance(DOS_ptrs, (BLOCK_M, 0))
        LSE_ptrs = tl.advance(LSE_ptrs, (BLOCK_M,))
        DELTAS_ptrs = tl.advance(DELTAS_ptrs, (BLOCK_M,))
        KBID_ptrs += N_BLOCKS

    tl.store(DK_ptrs, (dk * qk_scale).to(DK_ptrs.dtype.element_ty), boundary_check=(0,))
    tl.store(DV_ptrs, dv.to(DV_ptrs.dtype.element_ty), boundary_check=(0,))


@triton.jit
def _attn_bwd_dkdv(
        Q, K, V, DOS, DK, DV,
        KBID, LSE, DELTAS, qk_scale,
        Z: tl.constexpr,
        H: tl.constexpr,
        L: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N2: tl.constexpr,
        BLOCK_N: tl.constexpr,
        M_BLOCKS: tl.constexpr,
        N_BLOCKS: tl.constexpr,
        num_cores: tl.constexpr):

    N_factor = BLOCK_N2 // BLOCK_N
    NUM_BLOCKS_N = triton.cdiv(L, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_N * Z * H

    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_hz_idx = block_idx // NUM_BLOCKS_N
        task_n_idx = block_idx % NUM_BLOCKS_N
        task_n_idx2 = task_n_idx // N_factor

        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        stride_qz = H * L * D
        stride_qh = L * D
        stride_kbz = H * M_BLOCKS * N_BLOCKS
        stride_kbh = M_BLOCKS * N_BLOCKS
        stride_lsz = H * L
        stride_lsh = L
        # offset
        qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        kbid_offset = off_z.to(tl.int64) * stride_kbz + off_h.to(tl.int64) * stride_kbh
        lse_offset = off_z.to(tl.int64) * stride_lsz + off_h.to(tl.int64) * stride_lsh
        # ptr
        Q_block_ptr = tl.make_block_ptr(
                base=Q + qkv_offset,
                shape=(L, D),
                strides=(D, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_M, D),
                order=(1, 0))
        K_block_ptr = tl.make_block_ptr(
                base=K + qkv_offset,
                shape=(L, D),
                strides=(D, 1),
                offsets=(task_n_idx * BLOCK_N, 0),
                block_shape=(BLOCK_N, D),
                order=(1, 0))
        V_block_ptr = tl.make_block_ptr(
                base=V + qkv_offset,
                shape=(L, D),
                strides=(D, 1),
                offsets=(task_n_idx * BLOCK_N, 0),
                block_shape=(BLOCK_N, D),
                order=(1, 0))
        DOS_block_ptr = tl.make_block_ptr(
                base=DOS + qkv_offset,
                shape=(L, D),
                strides=(D, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_M, D),
                order=(1, 0))
        DK_block_ptr = tl.make_block_ptr(
                base=DK + qkv_offset,
                shape=(L, D),
                strides=(D, 1),
                offsets=(task_n_idx * BLOCK_N, 0),
                block_shape=(BLOCK_N, D),
                order=(1, 0))
        DV_block_ptr = tl.make_block_ptr(
                base=DV + qkv_offset,
                shape=(L, D),
                strides=(D, 1),
                offsets=(task_n_idx * BLOCK_N, 0),
                block_shape=(BLOCK_N, D),
                order=(1, 0))

        LSE_block_ptr = tl.make_block_ptr(
                base=LSE + lse_offset,
                shape=(L,),
                strides=(1,),
                offsets=(0,),
                block_shape=(BLOCK_M,),
                order=(0,))
        DELTAS_block_ptr = tl.make_block_ptr(
                base=DELTAS + lse_offset,
                shape=(L,),
                strides=(1,),
                offsets=(0,),
                block_shape=(BLOCK_M,),
                order=(0,))
        KBID_ptr = KBID + kbid_offset + task_n_idx2

        k = tl.load(K_block_ptr, boundary_check=(0,))
        v = tl.load(V_block_ptr, boundary_check=(0,))
        _attn_bwd_dkdv_inner(
            Q_ptrs=Q_block_ptr, k=k, v=v, DOS_ptrs=DOS_block_ptr, LSE_ptrs=LSE_block_ptr,
            DELTAS_ptrs=DELTAS_block_ptr, KBID_ptrs=KBID_ptr, DK_ptrs=DK_block_ptr, DV_ptrs=DV_block_ptr,
            qk_scale=qk_scale,
            L=L,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            N_BLOCKS=N_BLOCKS,
            )


@triton.jit
def _attn_fwd(
        Q, K, V, qk_scale,
        topk: tl.constexpr,
        LUT, LSE, OS,
        Z: tl.constexpr,
        H: tl.constexpr,
        L: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_M2: tl.constexpr,
        BLOCK_N: tl.constexpr,
        M_BLOCKS: tl.constexpr,
        num_cores: tl.constexpr):

    M_factor = BLOCK_M2 // BLOCK_M
    NUM_BLOCKS_M = triton.cdiv(L, BLOCK_M)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        task_m_idx2 = task_m_idx // M_factor

        off_z = task_hz_idx // H
        off_h = task_hz_idx % H

        stride_qz = H * L * D
        stride_qh = L * D
        qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        #lut_offset = off_z.to(tl.int64) * H * NUM_BLOCKS_M * topk + off_h.to(tl.int64) * NUM_BLOCKS_M * topk + task_m_idx.to(tl.int64) * topk
        lut_offset = off_z.to(tl.int64) * H * M_BLOCKS * topk + off_h.to(tl.int64) * M_BLOCKS * topk + task_m_idx2.to(tl.int64) * topk
        lse_offset = off_z.to(tl.int64) * H * L + off_h.to(tl.int64) * L

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)

        Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
        K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
        V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
        OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
        LUT_ptr = LUT + lut_offset
        LSE_ptrs = LSE + lse_offset + offs_m

        m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
        for m_block_idx in tl.range(topk):
            idx_n = tl.load(LUT_ptr + m_block_idx)
            n_mask = offs_n < L - idx_n * BLOCK_N

            k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])

            qk = tl.dot(q, tl.trans(k)) * (qk_scale * 1.4426950408889634)

            if L - idx_n * BLOCK_N < BLOCK_N:
                qk = tl.where(n_mask[None, :], qk, float("-inf"))


            v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(m_i, local_m)
            qk = qk - new_m[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - new_m)
            o_s = o_s * alpha[:, None]
            o_s += tl.dot(p.to(v.dtype), v)

            l_i = l_i * alpha + l_ij
            m_i = new_m


        o_s = o_s / l_i[:, None]
        tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)

        m_i += tl.math.log2(l_i)
        tl.store(LSE_ptrs, m_i, mask=offs_m < L)

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert k_block_id.is_contiguous() and lut.is_contiguous()

        # We recommend the following two settings
        #assert BLOCK_M == 64 or BLOCK_M == 128
        #assert BLOCK_N == 64

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {64, 128, 256}  # 注释用于泛化测试 HEAD_DIM_K

        if qk_scale is None:
            qk_scale = HEAD_DIM_K**-0.5
        
        o_s = torch.empty_like(q)
        lse = torch.empty(q.shape[0: 3], device=q.device, dtype=torch.float32)

        num_cores = aicore_num

        _attn_fwd[(num_cores,)](
            Q=q, K=k, V=v, qk_scale=qk_scale,
            topk=topk,
            LUT=lut, LSE=lse, OS=o_s,
            Z=q.shape[0],
            H=q.shape[1],
            L=q.shape[2],
            D=q.shape[3],
            BLOCK_M=64,
            BLOCK_M2=BLOCK_M,
            BLOCK_N=BLOCK_N,
            M_BLOCKS=triton.cdiv(q.shape[2], BLOCK_M),
            num_cores=num_cores)

        ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        # ctx.causal = causal
        
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        num_cube_cores = aicore_num
        num_vec_cores = vectorcore_num
        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, H, L, D = q.shape

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta_s = torch.empty_like(lse)

        BHL = B * H * L
        grid = (num_vec_cores, )
        _attn_bwd_preprocess[grid](
            o_s, do_s, delta_s,
            BHL, D
        )

        grid = (num_cube_cores, )
        _attn_bwd_dq[grid](
            q, k, v, lse, delta_s,
            do_s, dq, lut,
            ctx.qk_scale, ctx.topk,
            B, H,
            L, M_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            sub_BN = BLOCK_N//1,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5
        )
        #print(dq.shape)
        
        grid = (num_cube_cores, )
        _attn_bwd_dkdv[grid](
            Q=q, K=k, V=v, DOS=do_s, DK=dk, DV=dv,
            KBID=k_block_id, LSE=lse, DELTAS=delta_s, qk_scale=ctx.qk_scale,
            Z=q.shape[0],
            H=q.shape[1],
            L=q.shape[2],
            D=q.shape[3],
            BLOCK_M=BLOCK_M,
            BLOCK_N2=BLOCK_N,
            BLOCK_N=32,
            M_BLOCKS=M_BLOCKS,
            N_BLOCKS=N_BLOCKS,
            num_cores=num_cube_cores,
        )
        torch.npu.synchronize()
        return dq, dk, dv, None, None, None, None, None, None

