#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm066

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd /data/home/ad/zteng/vllm

# ============================================================
# Sim Recovery Control Test - Server (vLLM 0.6.6)
# Goals:
# - A1: optional --enable-chunked-prefill (slice prefill/recompute)
# - A1: --num-scheduler-steps (default fixed to 1)
# - A2: --max-num-batched-tokens / --max-num-seqs act as coarse "recovery budget"
# - Writes current_server_sig.json so client output naming auto encodes server group
# ============================================================

BASE_DIR="${BASE_DIR:-logs/recovery_ctrl}"
mkdir -p "${BASE_DIR}/server_logs" "${BASE_DIR}/summary"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED="${SERVED:-Qwen2.5-7B-Instruct}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

MEM="${MEM:-0.75}"
MAXLEN="${MAXLEN:-15000}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

PMODE="${PMODE:-recompute}"
SWAP_SPACE_GB="${SWAP_SPACE_GB:-0}"

ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"   # 0/1/true/false/yes/no
NUM_SCHED_STEPS="${NUM_SCHED_STEPS:-1}"                 # integer >= 1

CFG_TAG="${CFG_TAG:-simrc}"

READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
READY_POLL_S="${READY_POLL_S:-1}"
READY_PRINT_EVERY_S="${READY_PRINT_EVERY_S:-10}"

# -----------------------------
# normalize ENABLE_CHUNKED_PREFILL
# -----------------------------
case "${ENABLE_CHUNKED_PREFILL}" in
  1|true|TRUE|yes|YES) ENABLE_CHUNKED_PREFILL="1" ;;
  0|false|FALSE|no|NO|"") ENABLE_CHUNKED_PREFILL="0" ;;
  *)
    echo "[WARN] ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL} not recognized, treat as 0"
    ENABLE_CHUNKED_PREFILL="0"
    ;;
esac

# validate NUM_SCHED_STEPS
if ! [[ "${NUM_SCHED_STEPS}" =~ ^[0-9]+$ ]] || [[ "${NUM_SCHED_STEPS}" -lt 1 ]]; then
  echo "[WARN] NUM_SCHED_STEPS=${NUM_SCHED_STEPS} invalid, set to 1"
  NUM_SCHED_STEPS="1"
fi

# -----------------------------
# guards
# -----------------------------
if awk "BEGIN{exit !(${MEM} < 0.70)}"; then
  echo "[WARN] MEM=${MEM} < 0.70 is not allowed on this machine. Auto bump to 0.70"
  MEM="0.70"
fi

# IMPORTANT:
# - chunked prefill disabled: enforce MBT >= MAXLEN
# - chunked prefill enabled: allow MBT < MAXLEN (needed for sliced prefill/recompute)
if [[ "${ENABLE_CHUNKED_PREFILL}" == "0" ]]; then
  if [[ "${MAX_BATCH_TOKENS}" -lt "${MAXLEN}" ]]; then
    echo "[WARN] chunked_prefill disabled: MAX_BATCH_TOKENS(${MAX_BATCH_TOKENS}) < MAXLEN(${MAXLEN}), bump to ${MAXLEN}"
    MAX_BATCH_TOKENS="${MAXLEN}"
  fi
else
  echo "[INFO] chunked_prefill enabled: allow MAX_BATCH_TOKENS(${MAX_BATCH_TOKENS}) < MAXLEN(${MAXLEN})"
fi

# -----------------------------
# GPU selection: respect explicit CUDA_VISIBLE_DEVICES
# -----------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] Respect user CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES="1"
  echo "[INFO] CUDA_VISIBLE_DEVICES not set; default to ${CUDA_VISIBLE_DEVICES}"
fi

# ---- CUDA lib conflict guard (match your cluster issues) ----
unset CUDA_HOME CUDA_PATH
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/usr/local/cuda/lib64$' | paste -sd: -)"
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# -----------------------------
# sanity checks (torch + cuda)
# -----------------------------
python - <<'PY'
import sys
try:
    import torch
    print("[OK] torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("[OK] cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
    if torch.cuda.device_count() <= 0:
        print("[FATAL] torch sees 0 CUDA devices.", file=sys.stderr)
        sys.exit(2)
except Exception as e:
    print("[FATAL] torch import/cuda failed:", repr(e), file=sys.stderr)
    sys.exit(2)
PY

# -----------------------------
# audit-friendly tag
# -----------------------------
TAG="SimRC_cfg=${CFG_TAG}_served=${SERVED}_mem=${MEM}_maxlen=${MAXLEN}_mbt=${MAX_BATCH_TOKENS}_mseq=${MAX_NUM_SEQS}_pmode=${PMODE}_swap=${SWAP_SPACE_GB}_chunk=${ENABLE_CHUNKED_PREFILL}_ss=${NUM_SCHED_STEPS}_cvis=${CUDA_VISIBLE_DEVICES}"
LOG_FILE="${BASE_DIR}/server_logs/${TAG}.log"
PID_FILE="${BASE_DIR}/server_logs/${TAG}.pid"
CFG_FILE="${BASE_DIR}/summary/${TAG}.server_config.json"

# -----------------------------
# port check
# -----------------------------
if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[ERROR] Port ${PORT} already in use. Stop existing server first."
    exit 1
  fi
else
  echo "[WARN] lsof not found; skip port check."
fi

# -----------------------------
# write config snapshot
# -----------------------------
{
  echo "{"
  echo "  \"tag\": \"${TAG}\","
  echo "  \"cfg_tag\": \"${CFG_TAG}\","
  echo "  \"model\": \"${MODEL}\","
  echo "  \"served_model_name\": \"${SERVED}\","
  echo "  \"host\": \"${HOST}\","
  echo "  \"port\": ${PORT},"
  echo "  \"gpu_memory_utilization\": ${MEM},"
  echo "  \"max_model_len\": ${MAXLEN},"
  echo "  \"max_num_batched_tokens\": ${MAX_BATCH_TOKENS},"
  echo "  \"max_num_seqs\": ${MAX_NUM_SEQS},"
  echo "  \"preemption_mode\": \"${PMODE}\","
  echo "  \"swap_space_gb\": ${SWAP_SPACE_GB},"
  echo "  \"enable_chunked_prefill\": ${ENABLE_CHUNKED_PREFILL},"
  echo "  \"num_scheduler_steps\": ${NUM_SCHED_STEPS},"
  echo "  \"cuda_visible_devices\": \"${CUDA_VISIBLE_DEVICES}\","
  echo "  \"hf_home\": \"${HF_HOME}\","
  echo "  \"time\": \"$(date -Is)\""
  echo "}"
} > "${CFG_FILE}"

# -----------------------------
# write "current server signature" for client auto naming
# -----------------------------
SERVER_SIG_FILE="${SERVER_SIG_FILE:-${BASE_DIR}/current_server_sig.json}"
SERVER_SIG_STR="pmode=${PMODE}_chunk=${ENABLE_CHUNKED_PREFILL}_mbt=${MAX_BATCH_TOKENS}_mseq=${MAX_NUM_SEQS}_ss=${NUM_SCHED_STEPS}_mem=${MEM}_maxlen=${MAXLEN}_swap=${SWAP_SPACE_GB}_cvis=${CUDA_VISIBLE_DEVICES}_port=${PORT}"
{
  echo "{"
  echo "  \"server_sig_str\": \"${SERVER_SIG_STR}\","
  echo "  \"cfg_tag\": \"${CFG_TAG}\","
  echo "  \"pmode\": \"${PMODE}\","
  echo "  \"chunked\": ${ENABLE_CHUNKED_PREFILL},"
  echo "  \"mbt\": ${MAX_BATCH_TOKENS},"
  echo "  \"mseq\": ${MAX_NUM_SEQS},"
  echo "  \"ss\": ${NUM_SCHED_STEPS},"
  echo "  \"mem\": ${MEM},"
  echo "  \"maxlen\": ${MAXLEN},"
  echo "  \"swap_gb\": ${SWAP_SPACE_GB},"
  echo "  \"cuda_visible_devices\": \"${CUDA_VISIBLE_DEVICES}\","
  echo "  \"host\": \"${HOST}\","
  echo "  \"port\": ${PORT},"
  echo "  \"time\": \"$(date -Is)\""
  echo "}"
} > "${SERVER_SIG_FILE}"
echo "[INFO] wrote server signature: ${SERVER_SIG_FILE}"
echo "[INFO] server_sig_str=${SERVER_SIG_STR}"

echo "[INFO] Starting vLLM server..."
echo "[INFO] TAG: ${TAG}"
echo "[INFO] LOG: ${LOG_FILE}"
echo "[INFO] CFG: ${CFG_FILE}"

# -----------------------------
# build final command (print exact args)
# -----------------------------
CMD=(python -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --served-model-name "${SERVED}"
  --host "${HOST}" --port "${PORT}"
  --device cuda
  --dtype half
  --gpu-memory-utilization "${MEM}"
  --max-model-len "${MAXLEN}"
  --max-num-batched-tokens "${MAX_BATCH_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --preemption-mode "${PMODE}"
  --swap-space "${SWAP_SPACE_GB}"
  --num-scheduler-steps "${NUM_SCHED_STEPS}"
  --disable-log-requests
)

if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  CMD+=(--enable-chunked-prefill)
fi

{
  echo "[CMD] nohup ${CMD[*]} > ${LOG_FILE} 2>&1 &"
} | tee -a "${LOG_FILE}"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[INFO] PID: $(cat "${PID_FILE}")"

# -----------------------------
# readiness check
# -----------------------------
if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] curl not found; cannot do readiness check."
  exit 1
fi

echo "[INFO] Waiting for readiness via /v1/completions (timeout ${READY_TIMEOUT_S}s) ..."

READY_TMP="${BASE_DIR}/server_logs/${TAG}.ready_probe.json"
pid="$(cat "${PID_FILE}")"
last_print=0

for ((t=1; t<=READY_TIMEOUT_S; t+=READY_POLL_S)); do
  if ! ps -p "${pid}" >/dev/null 2>&1; then
    echo "[ERROR] server pid=${pid} exited early. Check ${LOG_FILE}"
    exit 1
  fi

  code="$(curl -sS --connect-timeout 1 --max-time 2 \
    -o "${READY_TMP}" -w "%{http_code}" \
    "http://${HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${SERVED}\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}" \
    2>/dev/null || echo "000")"

  if [[ "${code}" == "200" ]]; then
    echo "[INFO] Server ready."
    exit 0
  fi

  if (( t - last_print >= READY_PRINT_EVERY_S )); then
    last_print=$t
    if [[ "${code}" == "000" ]]; then
      echo "[INFO] waiting... t=${t}s http=000"
    else
      echo "[INFO] waiting... t=${t}s http=${code}"
      head -c 200 "${READY_TMP}" 2>/dev/null || true
      echo
    fi
  fi

  sleep "${READY_POLL_S}"
done

echo "[ERROR] Not ready in ${READY_TIMEOUT_S}s. Check ${LOG_FILE}"
exit 1
