#!/bin/bash
# Autonomous rebuttal launcher.
#
# Emits structured markers on stdout/driver.log that the harness Monitor tool
# can filter on:
#
#   [GPU_WAIT]  free=<MB>   -- polling step while waiting for GPU
#   [GPU_READY] free=<MB>   -- GPU freed enough to start
#   [STARTED]   exp=<key>   -- an experiment is about to start
#   [DONE]      exp=<key>   -- an experiment finished successfully
#   [FAILED]    exp=<key>   -- an experiment exited non-zero
#   [POSTPROC]  step=<name> -- figure / numbers generation step
#   [ALL_DONE]              -- everything finished, figures+numbers exported
#
# Re-running is safe: run_rebuttal_experiments.py skips (seed, model) pairs
# whose per-seed JSON already exists, so resumed runs make progress.

set -u

cd "$(dirname "$0")"
PY=/home/admin0/anaconda3/envs/zhangyue/bin/python
MIN_FREE_MB=${MIN_FREE_MB:-8000}
LOG_DIR=results/_launcher_logs
mkdir -p "$LOG_DIR"

ts() { date '+%F %T'; }

wait_for_gpu() {
  local waits=0
  while true; do
    local free_mb util
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [[ -z "$free_mb" ]]; then free_mb=0; fi
    if [[ "$free_mb" -ge "$MIN_FREE_MB" ]]; then
      echo "[$(ts)] [GPU_READY] free=${free_mb}MB util=${util}%"
      return 0
    fi
    echo "[$(ts)] [GPU_WAIT] free=${free_mb}MB util=${util}% need=${MIN_FREE_MB}MB"
    waits=$((waits + 1))
    # Backoff: first 10 waits -> 2 min, later -> 5 min, past 1h -> 10 min
    if   [[ $waits -le 10 ]]; then sleep 120
    elif [[ $waits -le 30 ]]; then sleep 300
    else sleep 600
    fi
  done
}

run_experiment() {
  local key="$1"
  local log="$LOG_DIR/${key}.log"
  echo "[$(ts)] [STARTED] exp=${key}"
  if $PY run_rebuttal_experiments.py --only "$key" >> "$log" 2>&1; then
    echo "[$(ts)] [DONE] exp=${key}"
  else
    echo "[$(ts)] [FAILED] exp=${key} log=${log}"
  fi
}

wait_for_gpu

# Order: cheapest first so data flows in quickly and incremental figures work.
EXPERIMENTS=(
  exp1b_har_classical
  exp1a_har_uci
  exp2_forecast
  exp4_ablation
  exp1c_har_pamap2
  exp5_sensitivity
  exp3_cross_forecast
  exp3_da_forecast
)

for exp in "${EXPERIMENTS[@]}"; do
  run_experiment "$exp"
done

echo "[$(ts)] [POSTPROC] step=make_latex_numbers"
$PY make_latex_numbers.py >> "$LOG_DIR/postproc.log" 2>&1 && \
  echo "[$(ts)] [DONE] step=make_latex_numbers" || \
  echo "[$(ts)] [FAILED] step=make_latex_numbers log=${LOG_DIR}/postproc.log"

echo "[$(ts)] [POSTPROC] step=plot_figures"
$PY plot_rebuttal_figures.py >> "$LOG_DIR/postproc.log" 2>&1 && \
  echo "[$(ts)] [DONE] step=plot_figures" || \
  echo "[$(ts)] [FAILED] step=plot_figures log=${LOG_DIR}/postproc.log"

echo "[$(ts)] [ALL_DONE] matrix=complete"
