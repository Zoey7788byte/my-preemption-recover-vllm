#!/usr/bin/env bash
set -euo pipefail

cd /data/home/ad/zteng/vllm

# =========================
# nohup wrapper
# =========================
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-logs/mot1/${RUN_TAG}}"
mkdir -p "${OUT_BASE}"

NOHUP_OUT="${OUT_BASE}/nohup.out"
NOHUP_PID="${OUT_BASE}/nohup.pid"

if [[ "${NOHUP_LAUNCHED:-0}" != "1" ]]; then
  nohup env NOHUP_LAUNCHED=1 RUN_TAG="${RUN_TAG}" OUT_BASE="${OUT_BASE}" bash "$0" \
    > "${NOHUP_OUT}" 2>&1 &
  echo $! > "${NOHUP_PID}"
  echo "[OK] launched nohup: pid=$(cat "${NOHUP_PID}") log=${NOHUP_OUT} out_base=${OUT_BASE}"
  exit 0
fi

# =========================
# env + conda
# =========================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm066
export PYTHONNOUSERSITE=1

# ---- CUDA lib conflict guard (match server) ----
unset CUDA_HOME CUDA_PATH
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/usr/local/cuda/lib64$' | paste -sd: -)"
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# =========================
# config
# =========================
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}/v1/completions}"
METRICS_URL="${METRICS_URL:-http://${HOST}:${PORT}/metrics}"
MODEL="${MODEL:-Qwen2.5-7B-Instruct}"

TRACE_PATH="${TRACE_PATH:-/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv}"
START_TS="${START_TS:-3063938.0}"
TRACE_WIN_S="${TRACE_WIN_S:-600}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-14500}"
REQSET_PATH="${REQSET_PATH:-${OUT_BASE}/reqset_trace.jsonl}"

PROMPTS_JSON="${PROMPTS_JSON:-/home/ad/zteng/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}"

LAMBDA_LIST="${LAMBDA_LIST:-0.2 0.3 0.35 0.4 0.45 0.5}"
DURATION_S="${DURATION_S:-180}"
MAX_OUTSTANDING="${MAX_OUTSTANDING:-128}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-0.2}"

SEED_REQSET="${SEED_REQSET:-0}"
SEED_ARRIVAL="${SEED_ARRIVAL:-0}"
BURST_WINDOW_S="${BURST_WINDOW_S:-2.0}"

SUMMARY_CSV="${SUMMARY_CSV:-${OUT_BASE}/summary.csv}"
PROGRESS_TXT="${PROGRESS_TXT:-${OUT_BASE}/progress.txt}"

echo "[INFO] OUT_BASE=${OUT_BASE}"
echo "[INFO] trace=${TRACE_PATH} start_ts=${START_TS} trace_win_s=${TRACE_WIN_S} max_total_tokens=${MAX_TOTAL_TOKENS}"
echo "[INFO] lambdas=${LAMBDA_LIST} duration_s=${DURATION_S} max_outstanding=${MAX_OUTSTANDING}"

# =========================
# helpers
# =========================
start_metrics_sampler() {
  local out_path="$1"
  local interval_s="$2"
  python3 -u - "$METRICS_URL" "$out_path" "$interval_s" <<'PY' &
import sys, time, requests
url, out, interval = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(out, "w", encoding="utf-8") as f:
    while True:
        ts = time.time()
        try:
            text = requests.get(url, timeout=2).text
        except Exception as e:
            text = f"#ERROR {repr(e)}"
        f.write(f"@@TS {ts:.6f}\n")
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
        f.flush()
        time.sleep(interval)
PY
}

write_progress() {
  local stage="$1"
  local lam="${2:-}"
  local extra="${3:-}"
  {
    echo "ts=$(date +%s)"
    echo "stage=${stage}"
    echo "lambda=${lam}"
    echo "out_base=${OUT_BASE}"
    echo "summary_csv=${SUMMARY_CSV}"
    echo "extra=${extra}"
  } > "${PROGRESS_TXT}"
}

wait_ready() {
  local timeout_s="${1:-600}"
  echo "[INFO] waiting server readiness (timeout ${timeout_s}s) ..."
  for ((i=1;i<=timeout_s;i++)); do
    code="$(curl -sS --connect-timeout 1 --max-time 2 \
      -o /tmp/vllm_ready_probe.json -w "%{http_code}" \
      "http://${HOST}:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}" \
      2>/dev/null || echo "000")"
    if [[ "${code}" == "200" ]]; then
      echo "[INFO] server ready."
      return 0
    fi
    if (( i % 10 == 0 )); then
      echo "[INFO] still waiting... t=${i}s http=${code}"
    fi
    sleep 1
  done
  echo "[FATAL] server not ready within ${timeout_s}s" >&2
  return 1
}

# =========================
# main
# =========================
write_progress "INIT" "" ""

wait_ready 600

# build reqset once
write_progress "BUILD_REQSET" "" "reqset_path=${REQSET_PATH}"
if [[ ! -s "${REQSET_PATH}" ]]; then
  python3 -u scripts/build_reqset_sweep_trace.py \
    "${TRACE_PATH}" "${START_TS}" "${TRACE_WIN_S}" "${MAX_TOTAL_TOKENS}" "${REQSET_PATH}" \
    | tee "${OUT_BASE}/build_reqset.log"
else
  echo "[INFO] reuse existing reqset: ${REQSET_PATH}"
fi

# run sweep
for LAM in ${LAMBDA_LIST}; do
  OUT_DIR="${OUT_BASE}/lambda_${LAM}"
  mkdir -p "${OUT_DIR}"
  write_progress "RUN_LAMBDA" "${LAM}" "out_dir=${OUT_DIR}"

  METR_RAW="${OUT_DIR}/metrics_raw.txt"
  CLIENT_CSV="${OUT_DIR}/client_results.csv"
  META_JSON="${OUT_DIR}/replay_meta.json"
  SEG_CSV="${OUT_DIR}/burst_segments.csv"
  CLIENT_LOG="${OUT_DIR}/client.log"

  start_metrics_sampler "${METR_RAW}" "${METRICS_INTERVAL_S}"
  SAMP_PID=$!

  t0="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"

  set +e
  python3 -u scripts/controlled_replay_sweep_trace.py \
    "${PROMPTS_JSON}" \
    "${TOKENIZER}" \
    "${REQSET_PATH}" \
    "${SEED_REQSET}" \
    "${BASE_URL}" \
    "${MODEL}" \
    "poisson" \
    "${LAM}" \
    "${DURATION_S}" \
    "1" \
    "${BURST_WINDOW_S}" \
    "${SEED_ARRIVAL}" \
    "${MAX_OUTSTANDING}" \
    "${MAX_TOTAL_TOKENS}" \
    "${CLIENT_CSV}" \
    "${META_JSON}" \
    "${SEG_CSV}" \
    > "${CLIENT_LOG}" 2>&1
  rc=$?
  set -e

  kill "${SAMP_PID}" >/dev/null 2>&1 || true
  wait "${SAMP_PID}" >/dev/null 2>&1 || true

  t1="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
  wall="$(python3 - <<PY
t0=float("${t0}"); t1=float("${t1}"); print(f"{(t1-t0):.6f}")
PY
)"

  if [[ "${rc}" -ne 0 ]]; then
    echo "[WARN] replay rc=${rc} lambda=${LAM} (still summarizing what exists)"
    write_progress "WARN_REPLAY_RC" "${LAM}" "rc=${rc}"
  fi

  write_progress "SUMMARIZE" "${LAM}" "wall=${wall}"
  python3 -u scripts/summarize_one_run_sweep_trace.py \
    "${OUT_DIR}" \
    "${LAM}" \
    "${DURATION_S}" \
    "${wall}" \
    "${META_JSON}" \
    "${SUMMARY_CSV}" \
    | tee -a "${OUT_BASE}/summarize.log"

  write_progress "DONE_LAMBDA" "${LAM}" ""
done

write_progress "DONE_ALL" "" ""
echo "[OK] sweep done: ${OUT_BASE}"
echo "[OK] summary: ${SUMMARY_CSV}"
