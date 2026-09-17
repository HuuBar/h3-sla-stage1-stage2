#!/bin/bash
# 目标机(80.5.9.128)启动 h3_train_hang 容器 (采样用) (2026-08-24)
# 与 108 的 start_h3_infer_108.sh 同款配置; 16 die 全量映射。
set -e

CTR=h3_train_hang
IMAGE=h3_train_hang:20260810

docker rm -f $CTR 2>/dev/null 2>/dev/null || true

DEVICES="--device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm"
for i in $(seq 0 15); do
  DEVICES="$DEVICES --device /dev/davinci$i"
done

docker run -d --name $CTR \
  --privileged --network host --ipc host --shm-size 500g \
  $DEVICES \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
  -v /etc/hixlep:/etc/hixlep \
  -v /mnt:/mnt \
  -v /home:/home \
  -v /var/log/npu:/usr/slog \
  $IMAGE sleep infinity

echo "[start] 容器已启动: $CTR"
sleep 10
docker ps --filter name=$CTR --format '{{.Names}} | {{.Status}}'

echo "[start] 验证容器内 NPU 与版本:"
docker exec $CTR bash -lc "npu-smi info 2>&1 | grep -cE '^\| [0-9]+ +[0-9]'; python3 -c 'import torch, torch_npu; print(\"torch\", torch.__version__, \"| torch_npu\", torch_npu.__version__, \"| devices\", torch_npu.npu.device_count())'"
