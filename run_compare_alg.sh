#之前跑的都是vllm-recompue

#对比算法1: vllm-swap

#server
CUDA_VISIBLE_DEVICES=2 MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=swap SWAP_SPACE_GB=32 ENABLE_CHUNKED_PREFILL=0 NUM_SCHED_STEPS=1 \
CFG_TAG=rc_base_swap BASE_DIR=logs/com_alg/vllm-swap \
bash scripts/vllm_swap_sever.sh


#client
RUN_TAG=$(date +%Y%m%d_%H%M%S) OUT_BASE=logs/com_alg/vllm-swap/${RUN_TAG} \
GATE_ENABLE=0 PHASE_SLICE_S=0 \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
bash scripts/Sim_recovery_control_test_client.sh