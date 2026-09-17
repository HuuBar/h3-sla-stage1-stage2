#!/bin/bash
# 130 机: newcont309-proj1000 起点, 主干+proj 联合续训, 全量数据过一遍 (~310 步)
# 数据: ir5k_gen_latent_cache_768p_stage2_full (4959 条) -> 4959/16 ≈ 310 步
# 超参: 主干 lr 1e-5 / proj_l 5e-5 (双 lr), task sft:train, 开放竞争 topk 0.05, zero3 16gpu offload
# 用法 (h3_train_hang 容器内):
#   bash /mnt/share/c50058431/c50058431/pkg_128_infer/scripts/train_backbone_full_130.sh
set -e
REPO=/mnt/share/c50058431/c50058431/minimax_dmd
DATA_BASE=/mnt/share/c50058431/c50058431/dataset/ir5k_gen_latent_cache_768p_stage2_full
MODEL_BASE=/mnt/share/c50058431/c50058431/FL2VA
CKPT=/mnt/share/c50058431/c50058431/minimax_dmd/models/train/merge/newcont309-proj1000.safetensors
OUT=$REPO/models/train/MiniMax-H3-T2VA-SLA-768p-backbone-full
CFG=$REPO/examples/minimax_h3/model_training/full/accelerate_config_zero3_16gpu_offload.yaml
TRAIN=$REPO/examples/minimax_h3/model_training/train.py
SAVE_STEPS=${SAVE_STEPS:-78}

[ -f "$CKPT" ] || { echo "起点不存在: $CKPT"; exit 1; }
[ -d "$DATA_BASE" ] || { echo "数据不存在: $DATA_BASE"; exit 1; }

N_FILES=$(ls "$DATA_BASE"/*.pth | wc -l)
echo "[bfull] 数据 $N_FILES 条 -> $((N_FILES/16)) 步/遍"

CKPT_LINK=/mnt/share/c50058431/c50058431/dataset/transformer_bfull_link.safetensors
ln -sf "$CKPT" "$CKPT_LINK"
MODEL_PATHS_DIT=$(python - "$CKPT_LINK" <<'PY'
import json, sys
print(json.dumps([[sys.argv[1]]]))
PY
)

export SLA_FORCE_OFF=1
cd $REPO

accelerate launch --config_file $CFG \
  --num_processes 16 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 29500 \
  $TRAIN \
  --dataset_base_path $DATA_BASE \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 768 --width 1344 --num_frames 124 \
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
  --save_steps $SAVE_STEPS \
  --enable_tensorboard_log \
  --task "sft:train"
