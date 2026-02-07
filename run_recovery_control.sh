
# 1) kill any server on port 8000
PID=$(lsof -tiTCP:8000 -sTCP:LISTEN || true)
if [ -n "$PID" ]; then
  echo "[INFO] Killing old server PID=$PID"
  kill "$PID" || true
  sleep 2
  PID2=$(lsof -tiTCP:8000 -sTCP:LISTEN || true)
  if [ -n "$PID2" ]; then
    echo "[INFO] Force killing PID=$PID2"
    kill -9 "$PID2" || true
  fi
else
  echo "[INFO] No server listening on :8000"
fi

#1111111=======
# Phase0 recovery observability quick test (trace replay).
cd /data/home/ad/zteng/vllm
CUDA_VISIBLE_DEVICES=2 \
VLLM_RECOVERY_OBS=0 \
MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=recompute SWAP_SPACE_GB=0 ENABLE_CHUNKED_PREFILL=0 NUM_SCHED_STEPS=1 \
CFG_TAG=phase0_baseline_simrc_ref \
bash scripts/phase0_recovery_server.sh


# Phase0 start client (trace replay; nohup wrapper inside script)
cd /data/home/ad/zteng/vllm

RUN_TAG=$(date +%Y%m%d_%H%M%S) \
GATE_ENABLE=0 PHASE_SLICE_S=0 \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
bash scripts/phase0_recovery_client.sh





#22222=======
#1) 启动 server（M2 micro swap-in）
CUDA_VISIBLE_DEVICES=2 \
MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=4096 MAX_NUM_SEQS=16 \
PMODE=recompute SWAP_SPACE_GB=0 ENABLE_CHUNKED_PREFILL=1 NUM_SCHED_STEPS=1 \
VLLM_RECOVERY_BUDGET=16 VLLM_RECOVERY_PHASE=1 \
CFG_TAG=phase1_m2_budget16 \
bash scripts/phase1_recovery_server.sh

#2）client
RUN_TAG=$(date +%Y%m%d_%H%M%S) \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
bash scripts/phase1_recovery_client.sh



#恢复备份
git apply logs/RecoveryGen/backups/phase1_save_20260207_121830/phase1_only.diff
