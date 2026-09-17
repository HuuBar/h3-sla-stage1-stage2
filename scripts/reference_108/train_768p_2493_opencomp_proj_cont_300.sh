#!/bin/bash
# ============================================================
# stage1 式线性分支再训 (单卡版, 2026-08-25, node-17-107)
# 起点: opencomp step-156 (主干 + proj_l 都是现在训练好的权重)
# 数据: 768p 2493 cache 再过一轮 (1 epoch = 156 步)
# 单卡原因: zero3 + bf16主干全冻结 + 仅fp32 proj_l有梯度 -> ds_id bucket 断言
#           (SKILL: sla-two-stage-training-20260817 坑2, 建议单卡; 8/15 stage1 即单卡)
# 方式: --proj-only 冻结主干 + --teacher-align 逐层对齐 (loss = 每层 o_sla vs o_full MSE)
# 开放竞争保持一致: SLA_FORCE_OFF=1, topk 0.05
# 保存: save_steps 40 -> step-40/80/120/156
# ============================================================
set -e

REPO=/mnt/share/c50058431/c50058431/minimax_dmd
DATA_BASE=/mnt/share/c50058431/c50058431/dataset/ir5k_gen_latent_cache_768p_stage2_2493_sub300
MODEL_BASE=/mnt/weight/MiniMax/MiniMax-H3/FL2VA
CKPT=/mnt/share/c50058431/c50058431/minimax_dmd/models/train/MiniMax-H3-T2VA-SLA-768p-2493-156-opencomp-proj/step-200-merged.safetensors
OUT=$REPO/models/train/MiniMax-H3-T2VA-SLA-768p-2493-156-opencomp-proj-cont
CFG=$REPO/examples/minimax_h3/model_training/full/accelerate_config_single_gpu.yaml
TRAIN=$REPO/examples/minimax_h3/model_training/train.py

[ -f "$CKPT" ] || { echo "[proj] 起点 ckpt 不存在: $CKPT"; exit 1; }

# use_sla 命中检查要求 model path 含 "transformer" (train.py:52), 软链满足
CKPT_LINK=/mnt/share/c50058431/c50058431/dataset/transformer_step200_merged_link.safetensors
ln -sf "$CKPT" "$CKPT_LINK"

MODEL_PATHS_DIT=$(python - "$CKPT_LINK" <<'PY'
import json, sys
print(json.dumps([[sys.argv[1]]]))
PY
)

export SLA_FORCE_OFF=1
export ASCEND_RT_VISIBLE_DEVICES=0
cd $REPO

echo "[$(date '+%F %T')] ==== SLA proj-only 再训(单卡): opencomp step-156 起点, teacher-align, lr=1e-4, 768p step-200 续训: 300 样本子集, 300 步 ===="
echo "[$(date '+%F %T')] model=$MODEL_PATHS_DIT"
echo "[$(date '+%F %T')] data=$DATA_BASE  SLA_FORCE_OFF=$SLA_FORCE_OFF"

accelerate launch --config_file $CFG \
  --num_processes 1 \
  $TRAIN \
  --dataset_base_path $DATA_BASE \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 768 \
  --width 1344 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_paths "$MODEL_PATHS_DIT" \
  --processor_path "$MODEL_BASE/processor" \
  --learning_rate 1e-4 \
  --enable_model_cpu_offload \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $OUT \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --use_sla \
  --sla_topk 0.05 \
  --sla_feature_map softmax \
  --sla_blkq 64 \
  --sla_blkk 64 \
  --proj-only \
  --teacher-align \
  --align-timesteps 50 \
  --save_steps 100 \
  --enable_tensorboard_log \
  --task "sft:train_align"
