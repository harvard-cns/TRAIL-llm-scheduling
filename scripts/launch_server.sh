#!/usr/bin/env bash
# Launch an SGLang server with one scheduling policy.
#
# Usage: scripts/launch_server.sh {fcfs|trail|trail_plus} [extra sglang.launch_server args]
#
#   fcfs        SGLang's default first-come, first-served scheduler
#   trail       TRAIL: SPRPT with limited preemption, driven by the released predictors
#               (DistilBERT on the prompt, then the layer-11 embedding MLP during decode)
#   trail_plus  TRAIL+: the same scheduler with oracle (exact) output lengths; requests
#               must carry ignore_eos=True and max_tokens = their true output length,
#               which scripts/benchmark.py does
#
# Env: MODEL (meta-llama/Meta-Llama-3-8B-Instruct), PORT (30000),
#      TRAIL_C (0)            limited-preemption threshold c; 0 turns preemption off
#                             (see README: on SGLang it measured best; the paper used 0.8)
#      PREDICT_EVERY (8)      refine the embedding prediction every K decode steps
#      ORACLE_MAE (0)         trail_plus only: add noise with this MAE (tokens)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${1:-}"
shift || true

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-30000}"
TRAIL_C="${TRAIL_C:-0}"
PREDICT_EVERY="${PREDICT_EVERY:-8}"
ORACLE_MAE="${ORACLE_MAE:-0}"

case "$POLICY" in
  fcfs)
    SCHEDULE_POLICY=fcfs
    ;;
  trail)
    SCHEDULE_POLICY=trail
    export SGLANG_TRAIL_INITIAL_PREDICTOR_DIR="$ROOT/predictors/initial_distilbert"
    export SGLANG_TRAIL_PREDICTOR_DIR="$ROOT/predictors/mlp_layer11"
    export SGLANG_TRAIL_PREDICT_EVERY="$PREDICT_EVERY"
    export SGLANG_TRAIL_PREEMPTION_LIMIT="$TRAIL_C"
    ;;
  trail_plus)
    SCHEDULE_POLICY=trail
    export SGLANG_TRAIL_ORACLE=1
    export SGLANG_TRAIL_ORACLE_MAE="$ORACLE_MAE"
    export SGLANG_TRAIL_PREEMPTION_LIMIT="$TRAIL_C"
    ;;
  *)
    echo "usage: $0 {fcfs|trail|trail_plus} [extra sglang args]" >&2
    exit 1
    ;;
esac

# flashinfer JIT-compiles some kernels on first launch and needs a CUDA toolkit.
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

echo "Launching $POLICY (schedule-policy=$SCHEDULE_POLICY, c=$TRAIL_C) on port $PORT"
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port "$PORT" \
  --schedule-policy "$SCHEDULE_POLICY" \
  "$@"
