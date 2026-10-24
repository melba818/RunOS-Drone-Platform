#!/usr/bin/env bash
# 停止 A6000 上的 OSTrack 跟踪服务，并确认 GPU1 显存已释放。
BASE=/srv/runhive
SVC=$BASE/services/ostrack
PIDF=$SVC/server.pid

PID=""
if [ -f "$PIDF" ]; then PID=$(cat $PIDF 2>/dev/null); fi
# pidfile 可能因为 setsid 的进程关系没写对，按命令行兜底再找一遍
PIDS=$(pgrep -f "services/ostrack/server.py" 2>/dev/null | tr '\n' ' ')
[ -n "$PID" ] && PIDS="$PIDS $PID"

if [ -n "$(echo $PIDS | tr -d ' ')" ]; then
  echo "killing: $PIDS"
  for p in $PIDS; do kill $p 2>/dev/null; done
  sleep 8
  LEFT=$(pgrep -f "services/ostrack/server.py" 2>/dev/null | tr '\n' ' ')
  if [ -n "$(echo $LEFT | tr -d ' ')" ]; then
    echo "force kill: $LEFT"
    for p in $LEFT; do kill -9 $p 2>/dev/null; done
    sleep 3
  fi
fi

rm -f $PIDF
echo "remaining procs: $(pgrep -af 'services/ostrack/server.py' 2>/dev/null || echo none)"
echo "--- gpu after stop ---"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
