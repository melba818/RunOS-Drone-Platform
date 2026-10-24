#!/usr/bin/env bash
# 在 A6000 上启动 OSTrack 跟踪服务（BF16 / GPU1 / 只绑 127.0.0.1:8022）。
# 由 .workbuddy/_ostrack_deploy.py --step start 远程调用，不要在 windows 本地执行。
#
# GPU 规划：GPU0 留给 VLX（不要动），OSTrack 独占 GPU1。
# 与 Qwen 互斥：自动标注模式启动 Qwen 前必须先 stop 本服务。
BASE=/srv/runhive
SVC=$BASE/services/ostrack
PY=$BASE/envs/vlxseek15/bin/python
PORT=${OSTRACK_PORT:-8022}
CKPT=$BASE/models/ostrack/vitb_256_mae_ce_32x4_ep300.pth.tar

if [ ! -x "$PY" ]; then
  echo "FATAL: venv python missing: $PY"
  exit 2
fi
if [ ! -f "$SVC/server.py" ]; then
  echo "FATAL: server.py missing: $SVC/server.py"
  exit 2
fi
if [ ! -f "$CKPT" ]; then
  echo "FATAL: checkpoint missing: $CKPT"
  exit 2
fi

# 启动前先看 GPU：确认 GPU1 空闲，且绝不碰 GPU0 上别人的作业
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

# 只允许占 GPU1
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OSTRACK_HOST=127.0.0.1
export OSTRACK_PORT=$PORT
export OSTRACK_CKPT=$CKPT

cd $SVC
rm -f $SVC/server.log $SVC/server.pid

# setsid 让服务脱离 SSH 会话，网络抖动不会把几分钟的加载掐掉
setsid nohup $PY $SVC/server.py > $SVC/server.log 2>&1 < /dev/null &
PID=$!
echo "pid=$PID"
echo $PID > $SVC/server.pid

sleep 5
echo "--- alive? ---"
ps -o pid,etime,cmd -p $PID | tail -n 1
echo "--- log head ---"
head -n 20 $SVC/server.log
