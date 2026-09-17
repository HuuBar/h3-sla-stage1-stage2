# H3 基于 diffsynth 训练 SLA(稀疏-线性注意力)方案

日期:2026-08-13
参考:
- 论文:arXiv:2509.24006 "SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention" (Tsinghua/Berkeley)
- 实现:gitcode.com/hyz22/MindSpeed-MM-SLA (Wan2.2 A14B, NPU/昇腾已适配)
- 本地副本:/mnt/a800_share/c50058431/sla_ref/(论文 PDF+txt、MindSpeed-MM-SLA 完整仓库)

=====================================================================
0. SLA 机制速览(论文核心)
=====================================================================
- 发现:DiT 注意力权重可分两类——少量大权重(高秩,~8%)+ 大量小权重(极低秩,~90%)
- 方案:注意力权重分三类:
    critical   (top 5% 块)  -> 稀疏 FlashAttention(块级 topk,仍 O(N^2) 但只算 5% 块)
    marginal   (其余)       -> 线性注意力 phi(Q)phi(K)^T V / rowsum,O(N) 代价
    negligible (bottom 10%) -> 直接跳过
- 融合:O = O_sparse + Proj(O_linear)
    Proj 是可学习线性层(head_dim -> head_dim),初始化为 0
    => 微调起点 = 纯稀疏注意力,与原模型输出接近,只需少量微调即收敛
- 实验:Wan2.1-1.3B,bs=64 微调 2000 步(< 预训练成本 0.1%),注意力计算减 95%,
    端到端 2.2x 加速,质量不掉;feature_map=softmax, block=64, top 5% / bottom 10%

=====================================================================
1. MindSpeed-MM-SLA 参考实现(关键文件)
=====================================================================
- mindspeed_mm/models/common/sla_core.py   (111 行)
    SparseLinearAttention(nn.Module):forward(q,k,v) -> O
    * get_block_map(q,k,topk_ratio,BLKQ,BLKK) 生成块级 topk LUT + sparse_map
    * sparse 部分走自研 Triton kernel(_attention),含前向/反向
    * linear 部分:feature_map(默认 softmax)+ kvsum 低秩 + proj_l(可训练 Linear)
    * 注:proj_l 的 init_weights_() 被注释掉(默认 kaiming 初始化),
      但我们移植时应改为显式置 0(论文要求,保证起点≈原注意力)
- mindspeed_mm/models/common/sla_kernel.py (499 行)
    * 已做 NPU 适配:import torch_npu + triton.ascend(容器实测可 import:
      triton 3.5.0 + triton_ascend 3.2.1 + torch_npu 2.10.0,aicore=24/vectorcore=48)
    * _attention(torch.autograd.Function):前向 _attn_fwd,反向 _attn_bwd_dq/_attn_bwd_dkdv
    * head_dim 支持 64/128/256(H3 是 128,满足)
- mindspeed_mm/models/common/sla_utils.py  (110 行)
    get_block_map:mean_pool(q/k 块) -> block score -> topk -> LUT + sparse_map
    (smooth-k: k - mean(k),SageAttention 技巧;910B1 注释说明已按 NPU 调过)
- mindspeed_mm/models/predictor/dits/wan_dit.py
    WanDiTBlock.__init__:if self.sla: sla_module = SparseLinearAttention(...)
    self_attn 传 sla=True+sla_module;cross_attn 仍正常注意力
- mindspeed_mm/models/common/attention.py
    FlashAttention.forward:if sla: output = sla_module(q, k, v)  # 替代 npu_fusion_attention
- mindspeed_mm/models/diffusion/wan_flow_match_scheduler.py
    q_sample 用 min/max_timestep_boundary 截断采样区间 -> 高低噪两阶段训练

=====================================================================
2. 高低噪训练(MindSpeed pretrain_high.sh / pretrain_low.sh)
=====================================================================
- 两套 json 的区别(pretrain_model_high.json vs low.json):
    high: max_timestep_boundary=0.417, min=0.0     -> 高噪声段(sigma 大)
    low:  max_timestep_boundary=1.0,  min=0.417    -> 低噪声段(sigma 小)
- scheduler q_sample:timestep_idx = randint(min_b, max_b) 在对应区间采样
- 其余训练参数相同:lr 1e-5, train-iters 500, MBS 1, 全参微调
- 数据:data_high.txt / data_low.txt 各指同一数据源不同目录(5000-1 / 5000-0 分片)
- 执行顺序:先 high 后 low,各自独立跑(各自 load/save 路径不同)

★ diffsynth 现成支持:loss.py 的 FlowMatchSFTMiniMaxH3AudioVideoLoss
  已内置 max_timestep_boundary/min_timestep_boundary(默认 1/0 = 全区间),
  train.py 只需把参数透传进 extra_inputs,loss 会自动截断采样区间。

=====================================================================
3. H3 与 Wan2.2 的结构差异(影响接入方式)
=====================================================================
- H3 attention 布局:varlen packed([total_tokens, H, D] 展平,按 cu_seqlens 分段)
    vs Wan 的 (B,H,L,D)。=> SLA 模块需按段调用:每段 reshape 成 (1,H,L,D)
- H3 无 cross-attention:文本经 token_refiner(2 层)后 index_add 拼进主序列,
    主序列 self-attn = 文本+视频+音频混排。=> SLA 会作用到全部 token(含文本),
    block topk 自然会把相关文本块选进 critical,风险可控;token_refiner 序列短(512)
    ,不值得 SLA,保留原注意力
- H3 head_dim=128(56 heads, hidden 5376)=> 满足 kernel 支持范围
- H3 双分支:video scheduler shift=12 / audio shift=3,loss 双 MSE。SLA 只改
    attention 内核,不动 scheduler/损失
- H3 主序列长(30K+ token)=> 正是 SLA 的目标场景

=====================================================================
4. 需要新增/修改的代码清单
=====================================================================
A. 新增 3 个文件(移植,基本照搬 MindSpeed):
   diffsynth/models/sla_core.py      (SparseLinearAttention;proj_l 显式置 0)
   diffsynth/models/sla_kernel.py    (NPU Triton kernel;独立文件,无 megatron 依赖)
   diffsynth/models/sla_utils.py     (get_block_map / mean_pool)

B. 修改 diffsynth/models/minimax_h3_dit.py:
   - MiniMaxH3Attention.__init__: 增加 sla/sla_topk/sla_feature_map/sla_blkq/sla_blkk
     参数;若 sla 则创建 SparseLinearAttention(head_dim=128,...) 实例
   - MiniMaxH3Attention.forward: 若 sla,对 cu_seqlens 每个段:
        seg_q = q[s:e].transpose(0,1).unsqueeze(0)  # (1,H,L,D)
        seg_out = sla_module(seg_q, seg_k, seg_v)
        out[s:e] = seg_out.squeeze(0).transpose(0,1)
     否则走原 _sdpa_varlen_attention
   - MiniMaxH3DiTBlock.__init__/forward: 透传 sla 配置
   - MiniMaxH3DiT.__init__: 增加 sla 配置参数;token_refiner 保持原样,
     主 blocks 传 sla=True
   - __init__.py 若需要导出新模块则补

C. 修改 examples/minimax_h3/model_training/train.py:
   - minimax_h3_parser() 增加 --sla/--sla-topk/--sla-feature-map/--sla-blkq/--sla-blkk
     /--max-timestep-boundary/--min-timestep-boundary
   - MiniMaxH3TrainingModule 构造 pipe 时把 sla 参数传进 model_paths/pipe 配置
     (需确认 MiniMaxH3Pipeline.from_pretrained 如何透传 dit 构造参数,
     可能在 ModelConfig 里加 sla 字段)
   - extra_inputs 追加 max_timestep_boundary/min_timestep_boundary,
     loss.py 已原生支持(第 66-94 行)
   - 注意 loss 里 timestep_id 对 video/audio 两个 scheduler 共用同一索引,
     两 scheduler 均 set_timesteps(1000, training=True) 时长度一致,OK

D. 权重/checkpoint:
   - 新增参数仅 proj_l(每主层 1 个 Linear 128x128,50 层 ≈ 0.82M 参数,可忽略)
   - 用原始权重(FL2VA/Ref2VA)初始化,新参数随机/置 0
   - save/load 走 diffsynth 现有机制,proj_l 会自动进入 state_dict

=====================================================================
5. 训练流程建议(两阶段)
=====================================================================
阶段 1(高噪):--max-timestep-boundary 0.417 --min-timestep-boundary 0
  从原始 FL2VA 权重起,全参微调,bs 视显存(参考 Wan:bs64/500 步,lr 1e-5)
阶段 2(低噪):--max-timestep-boundary 1.0 --min-timestep-boundary 0.417
  从阶段 1 的 checkpoint 起继续微调
- 数据:用刚下好的 IR5k/旧清单数据(音视频联合,32k 双声道)即可,
  论文说数据集与预训练一致即可,量不大(20K 条)也行
- 日志/ckpt 频率照旧(用户偏好频繁存盘)

=====================================================================
6. 风险与待验证点
=====================================================================
1. SLA kernel 在 h3_train_hang 已可 import,但未实际跑过 forward/backward:
   需先写最小单测(随机 q,k,v,对比 SLA vs 全注意力输出形状/数值范围,
   反向梯度能回传)再上训练
2. 推理侧:vllm-omni serve 用的是标准 flash attention,没有 SLA kernel。
   SLA 微调后推理时:
     a) 继续用全注意力推理(简单,但训练/推理 attention 不一致,需验证质量)
     b) 在 vllm-omni 也移植 SLA kernel(工作量大)
   先按 a) 验证,质量若崩再上 b)。论文 Fig2 显示 SLA 微调后质量与全注意力持平,
   有理由相信 a) 可行(proj_l 置 0 起点下,模型主要适配 sparse 部分)
3. gradient checkpointing 与自定义 autograd.Function 的兼容性:
   训练脚本默认开 checkpoint,需验证 SLA 段在 checkpoint 下不炸
4. varlen 分段下 SLA 的 topk 是"每段内"算的(每样本独立),与 Wan 全局一致;
   段长度差异大时 block 数差异大,topk 比例语义一致
5. 文本 token 混入主序列:SLA 的 block topk 可能漏掉部分文本块,
   若质量下降可考虑只对 video/audio 段做 SLA(实现稍复杂)
6. bf16 精度:kernel 内部转 bf16 计算(use_bf16=True),与 H3 bf16 训练一致

=====================================================================
7. 落地顺序
=====================================================================
1. [已验证] 拉论文+仓库到 /mnt/a800_share/c50058431/sla_ref/
2. [已验证] sla_kernel.py 在 h3_train_hang 单文件 import OK
3. 移植 3 个 sla 文件到 diffsynth/models/
4. 改 minimax_h3_dit.py 接入 sla(先写单测验证 forward/backward)
5. 改 train.py 透传参数 + 高低噪 boundary
6. 起阶段 1(高噪)训练,盯 tensorboard
7. 阶段 1 完 -> 阶段 2(低噪)
8. 微调后用 vllm-omni serve 验证(先全注意力,质量崩再移植 kernel 到推理)
