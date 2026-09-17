#!/bin/bash
# ============================================================
# 在 .114 (80.5.17.114) 上启动 h3_train_hang 训练容器 (2026-08-21)
# 挂载参考 .114 现有容器 lb_16pd (实测可用配置) + 旧 h3_train_hang 记录。
# ummu/uburma .114 上不存在, 按 ascend-container-dynamic-linking 结论跳过。
# 用法: 在 .114 上执行  bash /mnt/share/c50058431/c50058431/start_h3_train_114.sh
# ============================================================
set -e

CTR=h3_train_hang
IMAGE=h3_train_hang:20260810

docker rm -f $CTR 2>/dev/null || true

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
docker exec $CTR bash -lc "npu-smi info 2>&1 | head -6; python3 -c 'import torch, torch_npu; print(\"torch\", torch.__version__, \"| torch_npu\", torch_npu.__version__, \"| devices\", torch_npu.npu.device_count())'; python3 -c 'import deepspeed; print(\"deepspeed\", deepspeed.__version__)'; python3 -c 'import accelerate; print(\"accelerate\", accelerate.__version__)'; python3 -c 'import diffsynth; print(\"diffsynth\", diffsynth.__file__)'"
