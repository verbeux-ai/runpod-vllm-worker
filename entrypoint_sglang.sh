#!/bin/bash
# SGLang entrypoint: dispara modo vast.ai (com PyWorker) ou standalone.
set -e

echo "[entrypoint-sglang] env discovery: VAST_CONTAINERLABEL=${VAST_CONTAINERLABEL:-unset}"

if [ -n "$VAST_CONTAINERLABEL" ] || [ -n "$REPORT_ADDR" ]; then
  echo "[entrypoint-sglang] mode=vast.ai (SGLang + PyWorker)"

  # PyWorker openai handler espera SGLang em 127.0.0.1:30000 (default SGLang)
  export MODEL_PORT="${MODEL_PORT:-30000}"
  export BACKEND="${BACKEND:-openai}"
  export MODEL_NAME="${MODEL_NAME:-${SERVED_NAME:-${MODEL_ID##*/}}}"
  export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
  export SERVER_DIR="${SERVER_DIR:-/opt/vast-pyworker}"
  export ENV_PATH="${ENV_PATH:-/workspace/worker-env}"
  export MODEL_LOG="${MODEL_LOG:-/var/log/portal/sglang.log}"
  export WORKER_PORT="${WORKER_PORT:-3000}"
  export REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"

  # Patch worker.py do PyWorker pra usar nossa porta (default é 18000)
  if [ -f "$SERVER_DIR/workers/openai/worker.py" ]; then
    sed -i "s|MODEL_SERVER_PORT = 18000|MODEL_SERVER_PORT = ${MODEL_PORT}|g" "$SERVER_DIR/workers/openai/worker.py" || true
    sed -i "s|/var/log/portal/vllm.log|${MODEL_LOG}|g" "$SERVER_DIR/workers/openai/worker.py" || true
  fi

  mkdir -p "$(dirname "$MODEL_LOG")" "$WORKSPACE_DIR"
  : > "$MODEL_LOG"

  echo "[entrypoint-sglang] starting SGLang bg on :$MODEL_PORT, log=$MODEL_LOG"
  python3 -u /worker/handler_sglang.py --vast-bg >> "$MODEL_LOG" 2>&1 &
  SGL_PID=$!
  trap "kill $SGL_PID 2>/dev/null || true" EXIT

  echo "[entrypoint-sglang] starting PyWorker BACKEND=$BACKEND SERVER_DIR=$SERVER_DIR"
  exec bash "$SERVER_DIR/start_server.sh"

else
  echo "[entrypoint-sglang] mode=standalone (SGLang em foreground)"
  exec python3 -u /worker/handler_sglang.py --standalone
fi
