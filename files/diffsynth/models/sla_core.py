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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .sla_kernel import _attention
    from .sla_utils import get_block_map
except ImportError:  # 直接执行/独立测试时
    from sla_kernel import _attention
    from sla_utils import get_block_map


_FULL_ATTN_CHUNK = 512  # 全注意力分块: 避免 QK^T 物化 (15185^2*56*2B ≈ 26GB) 爆显存


def _full_attn(q, k, v, scale, chunk=_FULL_ATTN_CHUNK):
    """分块全注意力 (与 torch_sdpa 等价的逐块 SDPA), 供 teacher 对齐 / 判别实验用."""
    B, H, L, D = q.shape
    outs = []
    for s in range(0, L, chunk):
        outs.append(F.scaled_dot_product_attention(q[:, :, s:s + chunk], k, v, scale=scale))
    return torch.cat(outs, dim=2)


class _SparseAttentionWithTorchBackward(torch.autograd.Function):
    """Forward 走 NPU Triton kernel (原版, 已验证可用); backward 用纯 torch 算子重算。

    原因: _attn_bwd_dq / _attn_bwd_dkdv 在 NPU 上数值损坏 —— 隔离测试梯度 ~1e-9 (近乎为 0),
    真实训练中 _attentionBackward 的 dk 输出出现 nan -> 权重污染 -> step2 loss nan
    (detect_anomaly 实锤: Function '_attentionBackward' returned nan values in its 1th output)。
    backward 按标准 block-sparse softmax attention 公式重算 (对 lut 选中的 key 块重算
    scores -> softmax -> dv=P^T@do, dp=do@v^T, ds=P*(dp-delta), dq=ds@k, dk=ds^T@q),
    按 query 块分块控制内存。要求 BLKQ == BLKK (训练配置 64/64)。
    """

    @staticmethod
    def forward(ctx, q, k, v, sparse_map, lut, topk, BLKQ, BLKK):
        o_s = _attention.apply(q, k, v, sparse_map, lut, topk, BLKQ, BLKK)
        ctx.save_for_backward(q, k, v, lut)
        ctx.BLK = BLKQ
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        q, k, v, lut = ctx.saved_tensors
        BLK = ctx.BLK
        assert q.dim() == 4 and q.shape[-1] == k.shape[-1] == v.shape[-1]
        B, H, L, D = q.shape
        scale = D ** -0.5
        dev = q.device
        Lp = ((L + BLK - 1) // BLK) * BLK
        NB = Lp // BLK
        TK = lut.shape[-1]
        need_pad = Lp != L

        def _pad(t):
            return t if not need_pad else torch.nn.functional.pad(t, (0, 0, 0, Lp - L))

        # 保持 bf16 不整体 cast (OOM 规避); 只对分块切片转 fp32
        qp = _pad(q).reshape(H, Lp, D)
        kp = _pad(k).reshape(H, Lp, D)
        vp = _pad(v).reshape(H, Lp, D)
        dop = _pad(do_s).reshape(H, Lp, D)
        lut_h = lut.reshape(H, NB, TK)
        row_in_block = torch.arange(BLK, device=dev)  # (BLK,)

        dq = torch.zeros(H, Lp, D, dtype=torch.float32, device=dev)
        dk = torch.zeros(H, Lp, D, dtype=torch.float32, device=dev)
        dv = torch.zeros(H, Lp, D, dtype=torch.float32, device=dev)
        BLKD = BLK * D
        HG = 8   # 每批 head 数 (内存: 中间量 ~ x1/7)
        CH = 8   # 每 chunk 的 query 块数
        for h0 in range(0, H, HG):
            h1 = min(h0 + HG, H)
            Hg = h1 - h0
            qh = qp[h0:h1]; kh = kp[h0:h1]; vh = vp[h0:h1]; doh = dop[h0:h1]
            luth = lut_h[h0:h1]
            for s in range(0, NB, CH):
                e = min(s + CH, NB)
                ch = e - s
                idx = luth[:, s:e]  # (Hg, ch, TK) 选中的 key 块号
                idx3 = idx.reshape(Hg, ch * TK).unsqueeze(-1).expand(Hg, ch * TK, BLKD)
                k_sel = torch.gather(kh.reshape(Hg, NB, BLKD), 1, idx3).view(Hg, ch, TK, BLK, D).float()
                v_sel = torch.gather(vh.reshape(Hg, NB, BLKD), 1, idx3).view(Hg, ch, TK, BLK, D).float()
                qc = qh[:, s * BLK:e * BLK].view(Hg, ch, BLK, D).float()
                doc = doh[:, s * BLK:e * BLK].view(Hg, ch, BLK, D).float()

                sc = torch.einsum("hcbd,hctkd->hcbtk", qc, k_sel) * scale  # (Hg,ch,BLK,TK,BLK)
                if need_pad:
                    # 选中的 key 块里, 行号 >= L 的 padding 行 -> -inf
                    key_rows = idx.unsqueeze(-1) * BLK + row_in_block.view(1, 1, 1, BLK)  # (Hg,ch,TK,BLK)
                    sc = sc.masked_fill((key_rows >= L).unsqueeze(2), float("-inf"))
                # softmax 需跨全部选中 key (TK 块 × BLK 行)
                sc = sc.reshape(Hg, ch, BLK, TK * BLK)
                P = torch.softmax(sc, dim=-1).view(Hg, ch, BLK, TK, BLK)
                o_c = torch.einsum("hcbtk,hctkd->hcbd", P, v_sel)
                delta = (doc * o_c).sum(-1, keepdim=True).unsqueeze(-1)  # (Hg,ch,BLK,1,1) 广播到 (TK,BLK)
                dp = torch.einsum("hcbd,hctkd->hcbtk", doc, v_sel)
                ds = P * (dp - delta)
                dq_c = torch.einsum("hcbtk,hctkd->hcbd", ds, k_sel) * scale
                dv_sel = torch.einsum("hcbtk,hcbd->hctkd", P, doc)
                dk_sel = torch.einsum("hcbtk,hcbd->hctkd", ds, qc) * scale

                rows = (idx.reshape(Hg, ch * TK).unsqueeze(-1) * BLK + row_in_block.view(1, 1, BLK))
                # 各 head 的选中块不同: 展平到 (Hg*Lp) 维度, 索引加**组内** head 偏移
                # (目标张量是 dk[h0:h1] 切片, 只有 Hg*Lp 行; 用全局偏移 h0*Lp 会越界 -> MTE OOB)
                rows_flat = (rows + torch.arange(0, Hg, device=dev).view(Hg, 1, 1) * Lp).reshape(-1)
                dk.view(H, Lp, D)[h0:h1].reshape(Hg * Lp, D).index_add_(0, rows_flat, dk_sel.reshape(Hg * ch * TK * BLK, D))
                dv.view(H, Lp, D)[h0:h1].reshape(Hg * Lp, D).index_add_(0, rows_flat, dv_sel.reshape(Hg * ch * TK * BLK, D))
                dq[h0:h1, s * BLK:e * BLK] = dq_c.view(Hg, ch * BLK, D)

        dq = dq[:, :L].reshape(B, H, L, D).to(q.dtype)
        dk = dk[:, :L].reshape(B, H, L, D).to(k.dtype)
        dv = dv[:, :L].reshape(B, H, L, D).to(v.dtype)
        return dq, dk, dv, None, None, None, None, None


class SparseLinearAttention(nn.Module):
    def __init__(self, head_dim, topk, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
        '''
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)
        self.head_dim = head_dim
        # teacher 对齐模式 (阶段二): forward 时用原始 q/k/v 算全注意力 o_full,
        # loss = MSE(o_sla, o_full) 累加到 _align_loss_sum, 由训练脚本收集。
        # 通过环境变量 SLA_ALIGN_TEACHER=1 开启 (默认关闭, 不影响推理/既有训练)。
        self._align_loss_sum = None

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        # 零初始化: 训练起点 = 纯稀疏注意力 (输出 ≈ o_s), 更稳定. MindSpeed 原版注释掉了, 按论文保留.
        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def forward(self, q, k, v, return_sparsity=False, force_critical_mask=None):
        R'''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity (1 - selected ratio).
            force_critical_mask: (B, H, NQ, NK) int8/bool, 1 = key block forced visible for all query rows (column protection).
        '''
        dtype = q.dtype
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK,
                                                   force_critical_mask=force_critical_mask) # compress_kernel

        #print(lut.shape, real_topk)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype) 
        # forward 走 NPU Triton kernel; backward 用 torch 算子重算 (kernel 反向在 NPU 损坏)
        o_s = _SparseAttentionWithTorchBackward.apply(q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)
        #print(o_s.shape)
        # teacher 对齐用: feature_map 前的原始 qkv (与 o_s 同源, 与全注意力输入一致)
        q_raw, k_raw, v_raw = q, k, v
        q = self.feature_map_q(q).contiguous().to(self.dtype) # c_q
        k = self.feature_map_k(k).contiguous().to(self.dtype) # c_k
        def calc_linear(q, k, v):
            kvsum = k.transpose(-1, -2) @ v
            ksum = torch.sum(k, dim=-2, keepdim=True)
            return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))
        o_l = calc_linear(q, k, v)
        if os.environ.get("SLA_PROJ_ONLY"):
            self._o_l = o_l.detach()  # proj 前原始线性输出, 供训练脚本梯度模式重算

        # NPU 上无 'cuda' autocast 语义: proj_l 是 fp32 权重, 手动转 fp32 前向再转回
        o_l = self.proj_l(o_l.to(torch.float32)).to(self.dtype)
        # SLA_PROJ_ONLY=1 (proj_l 专用对齐训练): 主干冻结, o_s 梯度无用,
        # detach 后 sparse backward (torch 重算) 不再执行, 省显存省时间 (数学等价)
        if os.environ.get("SLA_PROJ_ONLY"):
            o_s = o_s.detach()
        o = (o_s + o_l).to(dtype)

        # SLA_ALIGN_TEACHER=1 (阶段二 主干+proj_l 联合): 每层累加 o_sla vs o_full
        # (全注意力, 分块防 QK^T 物化) 的 MSE。o_full 用原始 q/k/v (feature_map 前),
        # no_grad 作 teacher 目标; o_sla 带梯度 (主干 + proj_l 都能更新)。
        # 逐层反传模式: 每层 loss_i 立即 backward 释放本层图 (不攒 48 层),
        # 输出 o.detach() 传给下一层 (层间解耦, 各层独立对齐, 显存 O(单层))。
        if os.environ.get("SLA_ALIGN_TEACHER"):
            with torch.no_grad():
                o_full = _full_attn(q_raw, k_raw, v_raw, self.head_dim ** -0.5)
            loss_i = torch.nn.functional.mse_loss(o.float(), o_full.float())
            if self.training and o.requires_grad:
                loss_i.backward()  # 逐层立即反传, 图用完即释放
            self._align_loss_sum = loss_i.detach()
            o = o.detach()

        if return_sparsity:
            return o, 1.0 - real_topk / sparse_map.shape[-1]  # 真正稀疏度 (1 - 选中比例)
        else:
            return o




