#!/bin/bash
# proj_l 对齐训练启动器 (冒烟 100 步)
cd /mnt/a800_share/c50058431/minimax_dmd
mkdir -p /mnt/a800_share/c50058431/proj_l_align
exec python3 -u /mnt/a800_share/c50058431/tools/train_proj_only.py \
  --steps 100 --save-every 50 \
  --out /mnt/a800_share/c50058431/proj_l_align \
  > /mnt/a800_share/c50058431/proj_l_align/smoke.log 2>&1
