#!/bin/bash
# ============================================================
# 128 机: new-cont step-309 proj 续训 (16 卡尝试, 2026-09-03)
# 方式: --proj-only 冻结主干 + --teacher-align 逐层对齐 (和 opencomp-proj 156 同款)
# 起点: new-cont step-309 (SLA 开放竞争 topk 0.05)
# 数据: ir5k_gen_latent_cache_768p_stage2_full (全量 ~4959 条), repeat 1 -> ~310 步
# 超参: proj-only lr 1e-4, task sft:train_align, zero3 16gpu offload
# ⚠️ 已知风险: zero3 + bf16主干全冻结 + 仅fp32 proj_l有梯度 可能触发 ds_id 断言崩
#    (8/17 踩过, 当时改单卡; 8/25 opencomp-proj 16 卡跑通过) — 崩了告诉我, 给单卡版
# 用法 (h3_train_hang 容器内, 16 卡):
#   cd /mnt/share/c50058431/c50058431/minimax_dmd && \
#   SLA_FORCE_OFF=1 bash /mnt/share/c50058431/c50058431/pkg_128_infer/scripts/train_newcont_proj_16g_128.sh
# ============================================================
set -e

REPO=/mnt/share/c50058431/c50058431/minimax_dmd
DATA_BASE=/mnt/share/c50058431/c50058431/dataset/ir5k_gen_latent_cache_768p_stage2_full
MODEL_BASE=/mnt/share/c50058431/c50058431/FL2VA
CKPT=/mnt/share/c50058431/c50058431/minimax_dmd/models/train/MiniMax-H3-T2VA-SLA-768p-new-cont/step-309.safetensors
OUT=$REPO/models/train/MiniMax-H3-T2VA-SLA-768p-newcont-proj
CFG=$REPO/examples/minimax_h3/model_training/full/accelerate_config_zero3_16gpu_offload.yaml
TRAIN=$REPO/examples/minimax_h3/model_training/train.py
SAVE_STEPS=${SAVE_STEPS:-78}

[ -f "$CKPT" ] || { echo "[proj] 起点 ckpt 不存在: $CKPT"; exit 1; }
[ -d "$DATA_BASE" ] || { echo "[proj] 数据缓存不存在: $DATA_BASE"; exit 1; }

N_FILES=$(ls "$DATA_BASE"/*.pth | wc -l)
echo "[proj] 数据条数 $N_FILES -> $((N_FILES/16)) 步/遍 (repeat 1)"

CKPT_LINK=/mnt/share/c50058431/c50058431/dataset/transformer_newcont309_link.safetensors
ln -sf "$CKPT" "$CKPT_LINK"
MODEL_PATHS_DIT=$(python - "$CKPT_LINK" <<'PY'
import json, sys
print(json.dumps([[sys.argv[1]]]))
PY
)

export SLA_FORCE_OFF=1
cd $REPO

echo "[$(date '+%F %T')] ==== new-cont step-309 proj 续训(16卡): proj-only + teacher-align, topk 0.05 ===="
echo "[$(date '+%F %T')] data=$DATA_BASE ($N_FILES 条) model=$MODEL_PATHS_DIT"

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
  --learning_rate 1e-4 \
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
  --save_steps $SAVE_STEPS \
  --enable_tensorboard_log \
  --task "sft:train_align"
