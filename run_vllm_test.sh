#Motivation部分的目的：
  #在显存压力 + 真实到达/异质性下，系统确实会进入“抢占—恢复”常态路径，并且 tail latency 的主导因素来自恢复相关 stall 与振荡（而不是平均吞吐）

#Step0目的：展示当并发上升到某个阈值附近后，系统开始频繁抢占/恢复；平均吞吐未必掉很多，但 tail 急剧恶化。
# 下述没有发现抢占
#  --random-input-len 1024 \
#  --random-output-len 128 \
#Baseline Test
#服务器侧：

##############step0#############
#1) 杀掉占用 8000 的旧 server（无需 sudo）
PID=$(lsof -tiTCP:8000 -sTCP:LISTEN || true)
if [ -n "$PID" ]; then
  echo "[INFO] Killing old server PID=$PID"
  kill "$PID" || true
  sleep 2
  # 若还没退出，强杀
  PID2=$(lsof -tiTCP:8000 -sTCP:LISTEN || true)
  if [ -n "$PID2" ]; then
    echo "[INFO] Force killing PID=$PID2"
    kill -9 "$PID2" || true
  fi
else
  echo "[INFO] No server listening on :8000"
fi


#A寻找饱和点，扫不同的请求到达率，lambda
# 直接跑这个脚本就可以不区分server和client
conda activate vllm066
cd /data/home/ad/zteng/vllm
mkdir -p logs/sat_sweep

nohup env \
  RUN_TAG=$(date +%Y%m%d_%H%M%S) \
  CUDA_VISIBLE_DEVICES=2 \
  MEM_LIST="0.80" \
  LAMBDA_LIST="0.36 0.37 0.38 0.39 0.40 " \
  DURATION_S=180 \
  MAX_OUTSTANDING=128 \
  MODEL_REPO="Qwen/Qwen2.5-7B-Instruct" \
  SERVED_NAME="Qwen2.5-7B-Instruct" \
  PMODE=recompute SWAP_SPACE_GB=0 \
  MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
  bash scripts/step2_find_saturation_poisson_sweep.sh \
  > logs/sat_sweep/nohup_$(date +%Y%m%d_%H%M%S).out 2>&1 &
echo $! > logs/sat_sweep/nohup.pid


#step1:  扫λ找排队饱和点 真实trace跑真实 trace + Poisson 扫 真实 trace + Poisson
# 1) 启 server
conda activate vllm066
cd /data/home/ad/zteng/vllm

MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 PMODE=recompute \
SWAP_SPACE_GB=0 CUDA_VISIBLE_DEVICES=2 CFG_TAG=swap0_mem0.75 \
bash scripts/server_sweep_trace.sh


# 2)client: 新窗口 + 单点 lambda 探测
RUN_TAG=$(date +%Y%m%d_%H%M%S) \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 \
LAMBDA_LIST="0.55 0.60 0.65 0.70 0.75 0.80" \
DURATION_S=300 MAX_OUTSTANDING=256 MAX_TOTAL_TOKENS=14500 \
bash scripts/find_saturation_poisson_sweep_trace.sh


#step2: 验证mot1
#server
conda activate vllm066
cd /data/home/ad/zteng/vllm

MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 PMODE=recompute \
SWAP_SPACE_GB=0 CUDA_VISIBLE_DEVICES=2 CFG_TAG=swap0_mem0.75 \
bash scripts/mot1_server.sh


#client smooth
RUN_TAG=$(date +%Y%m%d_%H%M%S) \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 \
LAMBDA_LIST="0.75" \
DURATION_S=300 MAX_OUTSTANDING=256 MAX_TOTAL_TOKENS=14500 \
CFG_TAG=mot1_longwin_mem0.75_smooth \
bash scripts/find_saturation_poisson_sweep_trace.sh


#client burst b=5/10/20
RUN_TAG=$(date +%Y%m%d_%H%M%S) \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 \
LAMBDA_LIST="0.75" \
DURATION_S=300 MAX_OUTSTANDING=256 MAX_TOTAL_TOKENS=14500 \
MODE=burst BURST_B=20 \
CFG_TAG=mot1_longwin_mem0.75_burstb5 \
bash scripts/mot1_client_burst.sh



#step3: 验证mot2 抢占-恢复震荡，目标是：low段基本不抢占（让系统有恢复/清空队列的机会）
#high段稳定触发抢占但不至于，多个周期后能够看到preempt rate、waiting \Gpu mem的锯齿波
# 方案 A（最稳，先跑这个）
# λ_low = 0.60（你测到 0 抢占）
# λ_high = 0.80（你测到稳定抢占，且 wall_elapsed 还在可控范围附近）
# LOW_S = 120s，HIGH_S = 120s
# NUM_CYCLES = 8

# 这套最容易看到周期性“抢占—恢复—再抢占”。
# 方案 B（更强震荡，A 太弱时用）
# λ_low = 0.55
# λ_high = 0.85 或 0.90
# LOW_S = 150s，HIGH_S = 90s
# NUM_CYCLES = 10

# Mot2 的本质是“周期性负载波动”，用 λ 高低切换已经足够。
# 如果让 HIGH 段的“抢占更集中”，可以在 HIGH 段加 burstiness：
# LOW 段：MODE=poisson（或 BURST_B=1）
# HIGH 段：MODE=burst + BURST_B=10 或 20
# 这会更像“峰值阶段恢复成本集中化”，更容易形成锯齿波

#mot2 服务器
conda activate vllm066
cd /data/home/ad/zteng/vllm

MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 PMODE=recompute \
SWAP_SPACE_GB=0 CUDA_VISIBLE_DEVICES=2 CFG_TAG=swap0_mem0.75 \
bash scripts/mot2_server.sh

#方案A,高lambda\低lambda切换 client 
RUN_TAG=$(date +%Y%m%d_%H%M%S) \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 \
MAX_OUTSTANDING=256 MAX_TOTAL_TOKENS=14500 \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 \
LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
MODE_LOW=poisson MODE_HIGH=poisson \
METRICS_INTERVAL_S=0.5 \
METRICS_ENABLE=1 METRICS_TS_ENABLE=1 \
METRICS_KEYS="vllm:num_preemptions_total,vllm:num_requests_waiting,vllm:num_requests_running" \
CFG_TAG=mot2_final_mem0.75_longwin_l0.60_h0.85_180s_c8_mts0.5 \
bash scripts/mot2_oscillation_trace.sh


##############simple_recovery_process#############
#1）简单实现测试有效性
#可切分恢复控制--enable-chunked-prefill, 将prefill请求按max_num_batched_tokens进行切分
#在线优先的预算，--max-num-batched-tokens,每轮最多处理多少token(影响prefill chunk大小与每轮恢复推进量)
#这一步的验证目标：在相同压力与相同 trace 下，开启 chunked prefill 并下调 max-num-batched-tokens 是否能显著降低 TTFT/TPOT 的 p99/p99.9 与 stall gap 峰值，同时不显著降低稳态吞吐。

# 组0 Baseline 
#Server
CUDA_VISIBLE_DEVICES=2 MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=16384 MAX_NUM_SEQS=16 \
PMODE=recompute SWAP_SPACE_GB=0 ENABLE_CHUNKED_PREFILL=0 NUM_SCHED_STEPS=1 \
CFG_TAG=rc_base \
bash scripts/Sim_recovery_control_test_server.sh


#组0 client:不启用gate
RUN_TAG=$(date +%Y%m%d_%H%M%S) \
GATE_ENABLE=0 PHASE_SLICE_S=0 \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
bash scripts/Sim_recovery_control_test_client.sh


#组1 切片恢复
#server 启动切片恢复server: chunked+小预算
CUDA_VISIBLE_DEVICES=2 MEM=0.75 MAXLEN=15000 MAX_BATCH_TOKENS=4096 MAX_NUM_SEQS=16 \
PMODE=recompute SWAP_SPACE_GB=0 ENABLE_CHUNKED_PREFILL=1 NUM_SCHED_STEPS=1 \
CFG_TAG=rc_chunked_bt4096 \
bash scripts/Sim_recovery_control_test_server.sh

#client: gate仍关闭
RUN_TAG=$(date +%Y%m%d_%H%M%S) \
GATE_ENABLE=0 PHASE_SLICE_S=0 \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
bash scripts/Sim_recovery_control_test_client.sh


#组0和组1的差分是server侧的chunked prefill + 更小的max-num-batched-tokens

#组2 切片恢复+ 恢复窗口 (server 不变，用组1)
#client: 启用gate+slice检查
RUN_TAG=$(date +%Y%m%d_%H%M%S) \
GATE_ENABLE=1 GATE_GPU_PERC=0.92 GATE_PREEMPT_DELTA=1 GATE_T_S=2 \
PHASE_SLICE_S=10 \
LAMBDA_LOW=0.60 LAMBDA_HIGH=0.85 LOW_S=180 HIGH_S=180 NUM_CYCLES=8 \
TRACE_PATH=/home/ad/zteng/vllm/traces/BurstGPT_without_fails_1.csv \
START_TS=2032575.0 TRACE_WIN_S=300 MAX_TOTAL_TOKENS=14500 \
bash scripts/Sim_recovery_control_test_client.sh
