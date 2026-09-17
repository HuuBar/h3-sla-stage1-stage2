#!/bin/bash
# 130 机 (h3_train_hang 容器内): 单卡 proj 续训启动器 (默认 1000 步子集, 全量改 DATA_BASE)
cd /mnt/share/c50058431/c50058431/minimax_dmd || exit 1
export SLA_FORCE_OFF=1
export ASCEND_RT_VISIBLE_DEVICES=${NPU_IDX:-0}
nohup bash ../pkg_128_infer/scripts/train_newcont_proj_1g_128.sh > ../dataset/train_proj_1g.log 2>&1 &
echo "[launch-1g] started, log: ../dataset/train_proj_1g.log"
