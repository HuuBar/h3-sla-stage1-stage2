"""SLA 纯 PyTorch 参考实现 (与 MindSpeed Triton kernel 对拍用).

块级 topk mask + SDPA 稀疏分支 + 线性注意力分支, 完全可微.
仅用于数值正确性验证 / 无 kernel 环境的兜底, 训练默认走 sla_kernel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .sla_utils import get_block_map, mean_pool
except ImportError:
    from sla_utils import get_block_map, mean_pool


def sparse_attention_reference(q, k, v, sparse_map, BLKQ, BLKK, scale=None):
    """按块级 sparse_map 生成 token 级 mask, 走 SDPA."""
    B, H, L, D = q.shape
    if scale is None:
        scale = D ** -0.5
    # (B,H,NQ,NK) -> (B,H,NQ*BLKQ,NK*BLKK) -> 截断到 L; bool: True=attend
    mask = sparse_map.repeat_interleave(BLKQ, dim=-2).repeat_interleave(BLKK, dim=-1)
    mask = mask[..., :L, :L].bool()
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


class SparseLinearAttentionTorch(nn.Module):
    """纯 PyTorch 参考版. 接口与 sla_core.SparseLinearAttention 一致."""

    def __init__(self, head_dim, topk=0.05, feature_map="softmax", BLKQ=64, BLKK=64,
                 use_bf16=True, tie_feature_map_qk=True):
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if feature_map == "elu":
            self.feature_map_q = self.feature_map_k = (lambda x: F.elu(x) + 1)
        elif feature_map == "relu":
            self.feature_map_q = self.feature_map_k = nn.ReLU()
        elif feature_map == "softmax":
            self.feature_map_q = self.feature_map_k = (lambda x: F.softmax(x, dim=-1))
        else:
            raise NotImplementedError(f"Not supported feature map {feature_map}.")
        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q
        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def forward(self, q, k, v, return_sparsity=False):
        dtype = q.dtype
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)
        qb = q.to(self.dtype)
        kb = k.to(self.dtype)
        vb = v.to(self.dtype)
        o_s = sparse_attention_reference(qb, kb, vb, sparse_map, self.BLKQ, self.BLKK)
        qf = self.feature_map_q(qb).to(self.dtype)
        kf = self.feature_map_k(kb).to(self.dtype)
        kvsum = kf.transpose(-1, -2) @ vb
        ksum = torch.sum(kf, dim=-2, keepdim=True)
        o_l = (qf @ kvsum) / (1e-5 + (qf * ksum).sum(dim=-1, keepdim=True))
        o_l = self.proj_l(o_l.to(torch.float32)).to(self.dtype)
        o = (o_s + o_l).to(dtype)
        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        return o
