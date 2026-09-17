#!/bin/bash
# ============================================================
# 768p stage2 开放竞争续训 (2026-08-24, node-17-107)
# 起点: step-156.safetensors (MiniMax-H3-T2VA-SLA-768p-2493-156 产物)
# 再训: 1 遍 2493 条 / 16 卡 = 156 步
# 开放竞争: SLA_FORCE_OFF=1 (去掉文本/音频列保护, 纯全局 topk 0.05)
# 保存: save_steps 40 -> step-40/80/120/156 (zero3 gather, ~66GB/个)
# 超参: 主干 lr 1e-5 / proj_l 5e-5, task sft:train, zero3 16gpu offload
# 用法(107 容器内): cd REPO && SLA_FORCE_OFF=1 bash tools/train_768p_2493_opencomp_156.sh
# ============================================================
set -e

REPO=/mnt/share/c50058431/c50058431/minimax_dmd
DATA_BASE=/mnt/share/c50058431/c50058431/dataset/ir5k_gen_latent_cache_768p_stage2_2493
MODEL_BASE=/mnt/weight/MiniMax/MiniMax-H3/FL2VA
CKPT=/mnt/share/c50058431/c50058431/minimax_dmd/models/train/MiniMax-H3-T2VA-SLA-768p-2493-156/step-156.safetensors
OUT=$REPO/models/train/MiniMax-H3-T2VA-SLA-768p-2493-156-opencomp
CFG=$REPO/examples/minimax_h3/model_training/full/accelerate_config_zero3_16gpu_offload.yaml
TRAIN=$REPO/examples/minimax_h3/model_training/train.py

[ -f "$CKPT" ] || { echo "[train] 起点 ckpt 不存在: $CKPT"; exit 1; }

# 修复 (8/24): train.py 的 use_sla 命中检查要求 model path 含 "transformer" 字样
# (train.py:52 `hit = "transformer" in str(model_config.path)`), 否则模型按稠密构造.
# 用软链名满足检查 (与 auto_watch 的 transformer_resume_N 同款做法).
CKPT_LINK=/mnt/share/c50058431/c50058431/dataset/transformer_step156_link.safetensors
ln -sf "$CKPT" "$CKPT_LINK"

MODEL_PATHS_DIT=$(python - "$CKPT_LINK" <<'PY'
import json, sys
print(json.dumps([[sys.argv[1]]]))
PY
)

export SLA_FORCE_OFF=1
cd $REPO

echo "[$(date '+%F %T')] ==== 768p stage2 开放竞争续训: step-156 -> +156 步, topk 0.05 无列保护, save_steps 40 ===="
echo "[$(date '+%F %T')] data=$DATA_BASE"
echo "[$(date '+%F %T')] model=$MODEL_PATHS_DIT"
echo "[$(date '+%F %T')] SLA_FORCE_OFF=$SLA_FORCE_OFF (1=开放竞争, 无文本/音频列保护)"

accelerate launch --config_file $CFG \
  --num_processes 16 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 29500 \
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
  --learning_rate 1e-5 \
  --proj-lr 5e-5 \
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
  --save_steps 40 \
  --enable_tensorboard_log \
  --task "sft:train"
