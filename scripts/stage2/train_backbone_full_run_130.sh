#!/bin/bash
# 130 机 (h3_train_hang 容器内): 全量主干续训启动器 (只存最终档 SAVE_STEPS=310)
cd /mnt/share/c50058431/c50058431/minimax_dmd || exit 1
export SLA_FORCE_OFF=1
export SAVE_STEPS=310
nohup bash ../pkg_128_infer/scripts/train_backbone_full_130.sh > ../dataset/train_backbone_full.log 2>&1 &
echo "[launch-bfull] started, log: ../dataset/train_backbone_full.log"
