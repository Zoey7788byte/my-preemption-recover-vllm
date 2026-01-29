#!/usr/bin/env bash
set -euo pipefail

cd /data/home/ad/zteng/vllm

# ============================================================
# Sim Recovery Control Test - Client
# - C: mot2-style low/high lambda cycles
# - B: Admission Gate based on /metrics (client-side pause window)
# - Output naming auto encodes:
#   - server signature from logs/recovery_ctrl/current_server_sig.json
#   - gate signature from client envs
# - Directory naming uses rc_cycleXX_{low,high}_lam0p60_modepoisson (not legacy cycle_1_low_lambda_0.60)
# ============================================================

# =========================
# nohup wrapper
# =========================
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-logs/recovery_ctrl/${RUN_TAG}}"
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
# config: server endpoints
# =========================
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}/v1/completions}"
METRICS_URL="${METRICS_URL:-http://${HOST}:${PORT}/metrics}"
MODEL="${MODEL:-Qwen2.5-7B-Instruct}"

# server signature file written by server launcher
SERVER_SIG_FILE="${SERVER_SIG_FILE:-logs/recovery_ctrl/current_server_sig.json}"

# =========================
# config: trace replay
# =========================
TRACE_PATH="${TRACE_PATH:-/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv}"
START_TS="${START_TS:-2032575.0}"
TRACE_WIN_S="${TRACE_WIN_S:-300}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-14500}"

PROMPTS_JSON="${PROMPTS_JSON:-/home/ad/zteng/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}"

# =========================
# config: mot2 schedule
# =========================
MODE_LOW="${MODE_LOW:-poisson}"     # poisson | burst
MODE_HIGH="${MODE_HIGH:-${MODE_LOW}}"

LOW_LAMBDA="${LAMBDA_LOW:-${LOW_LAMBDA:-0.60}}"
HIGH_LAMBDA="${LAMBDA_HIGH:-${HIGH_LAMBDA:-0.85}}"
LOW_S="${LOW_S:-180}"
HIGH_S="${HIGH_S:-180}"
NUM_CYCLES="${NUM_CYCLES:-8}"

# burst params
BURST_B="${BURST_B:-4}"
BURST_WINDOW_S="${BURST_WINDOW_S:-2.0}"

# client load control
MAX_OUTSTANDING="${MAX_OUTSTANDING:-256}"

# =========================
# metrics sampling
# =========================
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-0.5}"
METRICS_CONTINUOUS="${METRICS_CONTINUOUS:-0}"  # 1 continuous, 0 per-phase

# =========================
# Admission Gate (simple control)
# =========================
GATE_ENABLE="${GATE_ENABLE:-0}"              # 0/1
GATE_GPU_PERC="${GATE_GPU_PERC:-0.92}"
GATE_PREEMPT_DELTA="${GATE_PREEMPT_DELTA:-1}"
GATE_WAITING_RISE="${GATE_WAITING_RISE:-0}"
GATE_WAITING_MIN="${GATE_WAITING_MIN:-0}"
GATE_T_S="${GATE_T_S:-2}"
GATE_MIN_INTERVAL_S="${GATE_MIN_INTERVAL_S:-0.5}"

# Optional: slice each phase into segments to allow mid-phase gating
PHASE_SLICE_S="${PHASE_SLICE_S:-0}"          # 0 disables; e.g., 5/10 enables

SEED_REQSET="${SEED_REQSET:-0}"
SEED_ARRIVAL_BASE="${SEED_ARRIVAL_BASE:-0}"

# =========================
# derive signatures for output naming
# =========================
SERVER_SIG_STR="unknown_server"
if [[ -s "${SERVER_SIG_FILE}" ]]; then
  SERVER_SIG_STR="$(python3 - <<PY
import json
p="${SERVER_SIG_FILE}"
try:
    d=json.load(open(p,"r",encoding="utf-8"))
    s=d.get("server_sig_str","unknown_server")
    s=s.replace("/","_").replace(" ","").replace(".","p")
    print(s)
except Exception:
    print("unknown_server")
PY
)"
else
  echo "[WARN] SERVER_SIG_FILE not found: ${SERVER_SIG_FILE} (use unknown_server)"
fi

GATE_SIG="gate0"
if [[ "${GATE_ENABLE}" == "1" ]]; then
  GATE_SIG="gate1_gpu${GATE_GPU_PERC}_dpre${GATE_PREEMPT_DELTA}_t${GATE_T_S}"
  if [[ "${PHASE_SLICE_S}" != "0" ]]; then
    GATE_SIG="${GATE_SIG}_slice${PHASE_SLICE_S}"
  fi
fi
GATE_SIG="${GATE_SIG//./p}"

# If OUT_BASE is default, rewrite it to include sigs
if [[ "${OUT_BASE}" == "logs/recovery_ctrl/${RUN_TAG}" ]]; then
  OUT_BASE="logs/recovery_ctrl/${RUN_TAG}_${SERVER_SIG_STR}_${GATE_SIG}"
  mkdir -p "${OUT_BASE}"
fi

# =========================
# derived paths
# =========================
REQSET_PATH="${REQSET_PATH:-${OUT_BASE}/reqset_trace.jsonl}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_BASE}/summary.csv}"
PROGRESS_TXT="${PROGRESS_TXT:-${OUT_BASE}/progress.txt}"
PHASE_FILE="${PHASE_FILE:-${OUT_BASE}/phase.txt}"
PHASE_LOG="${PHASE_LOG:-${OUT_BASE}/phase_changes.csv}"
GATE_STATE_FILE="${GATE_STATE_FILE:-${OUT_BASE}/gate_state.json}"

echo "[INFO] OUT_BASE=${OUT_BASE}"
echo "[INFO] server_sig_str=${SERVER_SIG_STR}"
echo "[INFO] gate_sig=${GATE_SIG}"
echo "[INFO] base_url=${BASE_URL} metrics_url=${METRICS_URL} model=${MODEL}"
echo "[INFO] trace=${TRACE_PATH} start_ts=${START_TS} trace_win_s=${TRACE_WIN_S} max_total_tokens=${MAX_TOTAL_TOKENS}"
echo "[INFO] low=${LOW_LAMBDA}(${MODE_LOW},${LOW_S}s) high=${HIGH_LAMBDA}(${MODE_HIGH},${HIGH_S}s) cycles=${NUM_CYCLES}"
echo "[INFO] burst_b=${BURST_B} burst_window_s=${BURST_WINDOW_S} max_outstanding=${MAX_OUTSTANDING}"
echo "[INFO] gate_enable=${GATE_ENABLE} gpu_thr=${GATE_GPU_PERC} dpre_thr=${GATE_PREEMPT_DELTA} waiting_rise=${GATE_WAITING_RISE} waiting_min=${GATE_WAITING_MIN} gate_t=${GATE_T_S} slice=${PHASE_SLICE_S}"

# =========================
# helpers
# =========================
write_progress() {
  local stage="$1"
  local extra="${2:-}"
  {
    echo "ts=$(date +%s)"
    echo "stage=${stage}"
    echo "out_base=${OUT_BASE}"
    echo "summary_csv=${SUMMARY_CSV}"
    echo "extra=${extra}"
  } > "${PROGRESS_TXT}"
}

log_phase_change() {
  local cycle_idx="$1"
  local phase_tag="$2"
  local lam="$3"
  local mode="$4"
  local ts
  ts="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
  if [[ ! -s "${PHASE_LOG}" ]]; then
    echo "ts,cycle,phase,lambda_rps,mode" > "${PHASE_LOG}"
  fi
  echo "${ts},${cycle_idx},${phase_tag},${lam},${mode}" >> "${PHASE_LOG}"
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

start_metrics_sampler() {
  local raw_out="$1"
  local csv_out="$2"
  local interval_s="$3"
  local phase_file="$4"
  local start_ts="$5"
  python3 -u - "$METRICS_URL" "$raw_out" "$csv_out" "$interval_s" "$phase_file" "$start_ts" <<'PY' &
import sys, time, requests, re
url, raw_out, csv_out, interval, phase_file, start_ts = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], float(sys.argv[6])

def read_phase():
    try:
        with open(phase_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def parse_metric(text, key):
    pat = re.compile(rf"^{re.escape(key)}\{{.*\}}\s+([-+]?\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return float("nan")
    return float("nan")

prev_ts = None
prev_pre = None

with open(raw_out, "w", encoding="utf-8") as fraw, open(csv_out, "w", encoding="utf-8") as fcsv:
    fcsv.write("ts,t_rel_s,phase,cycle,mode,lambda_rps,preempt_total,preempt_rate_per_s,req_waiting,req_running,gpu_cache_usage_perc\n")
    fcsv.flush()
    while True:
        ts = time.time()
        ph = read_phase()
        try:
            text = requests.get(url, timeout=2).text
        except Exception as e:
            text = f"#ERROR {repr(e)}"

        fraw.write(f"@@TS {ts:.6f}\n")
        fraw.write(text)
        if not text.endswith("\n"):
            fraw.write("\n")
        fraw.flush()

        pre = parse_metric(text, "vllm:num_preemptions_total")
        waiting = parse_metric(text, "vllm:num_requests_waiting")
        running = parse_metric(text, "vllm:num_requests_running")
        gpu_cache_perc = parse_metric(text, "vllm:gpu_cache_usage_perc")

        rate = float("nan")
        if prev_ts is not None and pre == pre:
            dt = max(1e-9, ts - prev_ts)
            if prev_pre is not None and prev_pre == prev_pre:
                rate = (pre - prev_pre) / dt
        prev_ts = ts
        prev_pre = pre

        cycle = ""
        mode = ""
        lam = ""
        phase_tag = ""
        for part in ph.split():
            if part.startswith("cycle="):
                cycle = part.split("=", 1)[1]
            elif part.startswith("phase="):
                phase_tag = part.split("=", 1)[1]
            elif part.startswith("lam="):
                lam = part.split("=", 1)[1]
            elif part.startswith("mode="):
                mode = part.split("=", 1)[1]

        t_rel = ts - start_ts
        fcsv.write(f"{ts:.6f},{t_rel:.6f},{phase_tag},{cycle},{mode},{lam},{pre},{rate},{waiting},{running},{gpu_cache_perc}\n")
        fcsv.flush()
        time.sleep(interval)
PY
}

gate_init() {
  if [[ "${GATE_ENABLE}" != "1" ]]; then
    return 0
  fi
  cat > "${GATE_STATE_FILE}" <<'JSON'
{"prev_ts":null,"prev_preempt":null,"prev_waiting":null}
JSON
  echo "[INFO] gate state initialized: ${GATE_STATE_FILE}"
}

gate_check_and_maybe_sleep() {
  if [[ "${GATE_ENABLE}" != "1" ]]; then
    return 0
  fi
  python3 - <<'PY' "${METRICS_URL}" "${GATE_STATE_FILE}" "${GATE_GPU_PERC}" "${GATE_PREEMPT_DELTA}" "${GATE_WAITING_RISE}" "${GATE_WAITING_MIN}" "${GATE_MIN_INTERVAL_S}" "${GATE_T_S}"
import json, time, re, sys
import requests

url = sys.argv[1]
state_path = sys.argv[2]
gpu_thr = float(sys.argv[3])
pre_delta_thr = float(sys.argv[4])
waiting_rise = int(sys.argv[5])
waiting_min = float(sys.argv[6])
min_interval = float(sys.argv[7])
gate_t = float(sys.argv[8])

def parse_metric(text, key):
    pat = re.compile(rf"^{re.escape(key)}\{{.*\}}\s+([-+]?\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return float("nan")
    return float("nan")

try:
    st = json.load(open(state_path, "r", encoding="utf-8"))
except Exception:
    st = {"prev_ts": None, "prev_preempt": None, "prev_waiting": None}

now = time.time()
prev_ts = st.get("prev_ts", None)
if prev_ts is not None and now - float(prev_ts) < min_interval:
    sys.exit(0)

try:
    text = requests.get(url, timeout=2).text
except Exception:
    st["prev_ts"] = now
    json.dump(st, open(state_path, "w", encoding="utf-8"))
    sys.exit(0)

pre = parse_metric(text, "vllm:num_preemptions_total")
waiting = parse_metric(text, "vllm:num_requests_waiting")
gpu = parse_metric(text, "vllm:gpu_cache_usage_perc")

prev_pre = st.get("prev_preempt", None)
prev_wait = st.get("prev_waiting", None)

trigger = False
reasons = []

if gpu == gpu and gpu > gpu_thr:
    trigger = True
    reasons.append(f"gpu={gpu:.3f}>{gpu_thr}")

if pre == pre and prev_pre is not None:
    dpre = pre - float(prev_pre)
    if dpre >= pre_delta_thr:
        trigger = True
        reasons.append(f"dpre={dpre:.0f}>={pre_delta_thr}")

if waiting == waiting:
    if waiting_min > 0 and waiting >= waiting_min:
        trigger = True
        reasons.append(f"waiting={waiting:.0f}>={waiting_min}")
    if waiting_rise == 1 and prev_wait is not None and waiting > float(prev_wait):
        trigger = True
        reasons.append(f"waiting_rise {float(prev_wait):.0f}->{waiting:.0f}")

st["prev_ts"] = now
if pre == pre:
    st["prev_preempt"] = pre
if waiting == waiting:
    st["prev_waiting"] = waiting
json.dump(st, open(state_path, "w", encoding="utf-8"))

if trigger:
    print("[GATE] trigger:", "; ".join(reasons))
    time.sleep(gate_t)
PY
}

run_one_segment() {
  local out_dir="$1"
  local lam="$2"
  local mode="$3"
  local dur_s="$4"
  local seed_arrival="$5"

  mkdir -p "${out_dir}"

  local client_csv="${out_dir}/client_results.csv"
  local meta_json="${out_dir}/replay_meta.json"
  local seg_csv="${out_dir}/burst_segments.csv"
  local client_log="${out_dir}/client.log"

  set +e
  python3 -u scripts/controlled_replay_sweep_trace.py \
    "${PROMPTS_JSON}" \
    "${TOKENIZER}" \
    "${REQSET_PATH}" \
    "${SEED_REQSET}" \
    "${BASE_URL}" \
    "${MODEL}" \
    "${mode}" \
    "${lam}" \
    "${dur_s}" \
    "${BURST_B}" \
    "${BURST_WINDOW_S}" \
    "${seed_arrival}" \
    "${MAX_OUTSTANDING}" \
    "${MAX_TOTAL_TOKENS}" \
    "${client_csv}" \
    "${meta_json}" \
    "${seg_csv}" \
    > "${client_log}" 2>&1
  local rc=$?
  set -e
  echo "${rc}"
}

summarize_one_segment() {
  local out_dir="$1"
  local lam="$2"
  local dur_s="$3"
  local wall="$4"
  local meta_json="${out_dir}/replay_meta.json"
  python3 -u scripts/summarize_one_run_sweep_trace.py \
    "${out_dir}" \
    "${lam}" \
    "${dur_s}" \
    "${wall}" \
    "${meta_json}" \
    "${SUMMARY_CSV}"
}

run_phase() {
  local cycle_idx="$1"
  local phase_tag="$2"
  local lam="$3"
  local seed_arrival="$4"
  local mode="$5"
  local dur_s="$6"

  local lam_tag
  lam_tag="$(python3 - <<PY
x=float("${lam}")
print(f"{x:.2f}".replace(".","p"))
PY
)"
  local out_dir="${OUT_BASE}/rc_cycle$(printf "%02d" "${cycle_idx}")_${phase_tag}_lam${lam_tag}_mode${mode}"
  mkdir -p "${out_dir}"

  echo "cycle=${cycle_idx} phase=${phase_tag} lam=${lam} mode=${mode}" > "${PHASE_FILE}"

  # gate before phase (and before each slice if slicing enabled)
  gate_check_and_maybe_sleep

  local metr_raw="${out_dir}/metrics_raw.txt"
  local metr_csv="${out_dir}/metrics_timeseries.csv"
  local samp_pid=""
  if [[ "${METRICS_CONTINUOUS}" == "0" ]]; then
    start_metrics_sampler "${metr_raw}" "${metr_csv}" "${METRICS_INTERVAL_S}" "${PHASE_FILE}" "${RUN_START_TS}"
    samp_pid=$!
  fi

  local rc=0

  if [[ "${PHASE_SLICE_S}" == "0" ]]; then
    local t0 t1
    t0="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
    rc="$(run_one_segment "${out_dir}" "${lam}" "${mode}" "${dur_s}" "${seed_arrival}")"
    t1="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
    local wall
    wall="$(python3 - <<PY
t0=float("${t0}"); t1=float("${t1}"); print(f"{(t1-t0):.6f}")
PY
)"
    summarize_one_segment "${out_dir}" "${lam}" "${dur_s}" "${wall}" | tee -a "${OUT_BASE}/summarize.log"
  else
    local remaining="${dur_s}"
    local k=0
    while (( remaining > 0 )); do
      k=$((k+1))
      local seg_dur="${PHASE_SLICE_S}"
      if (( seg_dur > remaining )); then
        seg_dur="${remaining}"
      fi

      gate_check_and_maybe_sleep

      local sdir="${out_dir}/slice_$(printf "%02d" "${k}")"
      local s0 s1 swall
      s0="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
      local rck
      rck="$(run_one_segment "${sdir}" "${lam}" "${mode}" "${seg_dur}" "$((seed_arrival + k))")"
      if [[ "${rck}" -ne 0 ]]; then
        rc="${rck}"
        echo "[WARN] slice rc=${rck} cycle=${cycle_idx} phase=${phase_tag} k=${k}" | tee -a "${OUT_BASE}/warn.log"
      fi
      s1="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"
      swall="$(python3 - <<PY
s0=float("${s0}"); s1=float("${s1}"); print(f"{(s1-s0):.6f}")
PY
)"
      summarize_one_segment "${sdir}" "${lam}" "${seg_dur}" "${swall}" | tee -a "${OUT_BASE}/summarize.log"

      remaining=$((remaining - seg_dur))
    done
  fi

  if [[ "${METRICS_CONTINUOUS}" == "0" ]]; then
    kill "${samp_pid}" >/dev/null 2>&1 || true
    wait "${samp_pid}" >/dev/null 2>&1 || true
  fi

  if [[ "${rc}" -ne 0 ]]; then
    echo "[WARN] phase finished with rc=${rc}: cycle=${cycle_idx} phase=${phase_tag} lam=${lam}" | tee -a "${OUT_BASE}/warn.log"
  fi
}

# =========================
# main
# =========================
write_progress "INIT" ""

RUN_START_TS="$(python3 - <<'PY'
import time; print(f"{time.time():.6f}")
PY
)"

wait_ready 600
gate_init

# build reqset once
write_progress "BUILD_REQSET" "reqset_path=${REQSET_PATH}"
if [[ ! -s "${REQSET_PATH}" ]]; then
  python3 -u scripts/build_reqset_sweep_trace.py \
    "${TRACE_PATH}" "${START_TS}" "${TRACE_WIN_S}" "${MAX_TOTAL_TOKENS}" "${REQSET_PATH}" \
    | tee "${OUT_BASE}/build_reqset.log"
else
  echo "[INFO] reuse existing reqset: ${REQSET_PATH}"
fi

# optional continuous metrics sampler
if [[ "${METRICS_CONTINUOUS}" == "1" ]]; then
  METR_CONT_RAW="${OUT_BASE}/metrics_raw.txt"
  METR_CONT_CSV="${OUT_BASE}/metrics_timeseries.csv"
  echo "cycle=0 phase=init lam=0 mode=${MODE_LOW}" > "${PHASE_FILE}"
  start_metrics_sampler "${METR_CONT_RAW}" "${METR_CONT_CSV}" "${METRICS_INTERVAL_S}" "${PHASE_FILE}" "${RUN_START_TS}"
  CONT_PID=$!
  echo "[INFO] metrics continuous: ${METR_CONT_CSV} (pid=${CONT_PID})"
fi

for ((c=1; c<=NUM_CYCLES; c++)); do
  write_progress "CYCLE_LOW" "cycle=${c} lam=${LOW_LAMBDA} mode=${MODE_LOW} dur_s=${LOW_S}"
  log_phase_change "${c}" "low" "${LOW_LAMBDA}" "${MODE_LOW}"
  run_phase "${c}" "low" "${LOW_LAMBDA}" "$((SEED_ARRIVAL_BASE + c*2))" "${MODE_LOW}" "${LOW_S}"

  write_progress "CYCLE_HIGH" "cycle=${c} lam=${HIGH_LAMBDA} mode=${MODE_HIGH} dur_s=${HIGH_S}"
  log_phase_change "${c}" "high" "${HIGH_LAMBDA}" "${MODE_HIGH}"
  run_phase "${c}" "high" "${HIGH_LAMBDA}" "$((SEED_ARRIVAL_BASE + c*2 + 1))" "${MODE_HIGH}" "${HIGH_S}"
done

if [[ "${METRICS_CONTINUOUS}" == "1" ]]; then
  kill "${CONT_PID}" >/dev/null 2>&1 || true
  wait "${CONT_PID}" >/dev/null 2>&1 || true
fi

write_progress "DONE_ALL" ""
echo "[OK] recovery-control run done: ${OUT_BASE}"
echo "[OK] summary: ${SUMMARY_CSV}"
