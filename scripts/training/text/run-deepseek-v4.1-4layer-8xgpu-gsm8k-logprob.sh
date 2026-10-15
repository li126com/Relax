#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Online two-step counterpart of Miles' H800 all-on GSM8K run, with per-token loss.
# Supports direct bash launch via the standard local.sh entrypoint, or ray-job.sh.
# MODEL_DIR, DATA_DIR and SAVE_DIR are required. Choose a fresh EXP_NAME for a new baseline.
# DRY_RUN=1 prints argv/runtime env without starting Ray or a model.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/DeepSeek-V4.1-4layer}"
EXP_NAME="${EXP_NAME:-relax-v41-all-on-online-gsm8k-response256-per-token-2step}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/deepseek-v41-gsm8k-logprob}"
USE_CLEARML="${USE_CLEARML:-1}"
source "${RELAX_ROOT}/scripts/models/deepseek-v4.1-4layer.sh"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${HF_CKPT}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}/checkpoints"
    --save "${SAVE_DIR}/${EXP_NAME}/checkpoints"
    --save-interval 20
    --max-actor-ckpt-to-keep 2
    --dump-details "${SAVE_DIR}/${EXP_NAME}/dump_details"
    --custom-megatron-init-path scripts.tools.deepseek_v41_logprob_compare.initialize_reduced_experiment
)
ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/gsm8k/train.parquet"
    --input-key messages
    --label-key label
    --apply-chat-template
    --apply-chat-template-kwargs '{}'
    --rollout-shuffle
    --rollout-seed 42
    --rm-type math
    --num-rollout 2
    --rollout-batch-size 16
    --n-samples-per-prompt 8
    --rollout-max-response-len 256
    --rollout-temperature 0.8
    --rollout-top-p 1
    --rollout-top-k -1
    --global-batch-size 128
    --balance-data
    --use-rollout-routing-replay
)
PERF_ARGS=(
    --qkv-format bshd
    --seq-length 4096
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --context-parallel-size 2
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --allgather-cp
    # Relax rounds to 2*CP*TP*multiplier; Miles rounds to TP*128.
    # At TP2/CP2, 32 here gives the same 256-token global alignment as Miles.
    --data-pad-size-multiplier 32
    --calculate-per-token-loss
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --micro-batch-size 1
    --max-tokens-per-gpu 2048
    --log-probs-max-tokens-per-gpu 2048
)
GRPO_ARGS=(
    --advantage-estimator grpo
    --loss-type policy_loss
    --kl-coef 0.00
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --no-pin-cpu-grads
    --no-pin-cpu-params
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-tp-size 8
    --sglang-dp-size 1
    --sglang-ep-size 8
    --sglang-load-format auto
    --sglang-attention-backend dsv4
    --sglang-moe-runner-backend auto
    --sglang-mem-fraction-static 0.6
    --sglang-max-running-requests 128
    --sglang-disable-cuda-graph
    --sglang-disable-radix-cache
    --sglang-watchdog-timeout 3600
)
METRICS_ARGS=()
if [ "${USE_CLEARML}" = 1 ]; then
    METRICS_ARGS+=(
        --use-clearml
        --use-metrics-service
        --tb-project-name "${PROJECT_NAME}"
        --tb-experiment-name "${EXP_NAME}-${now}"
    )
fi
MISC_ARGS=(
    --bf16
    --transformer-impl transformer_engine
    --seed 1234
    --deterministic-mode
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --trust-remote-code
    --use-health-check
    --update-weight-buffer-size 1073741824
)
TRAIN_ARGS=(
    --resource '{"actor": [1, 8], "rollout": [1, 8]}'
    --num-gpus-per-node 8
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
    --offload-train
    --offload-rollout
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${PERF_ARGS[@]}"
    "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${SGLANG_ARGS[@]}" "${METRICS_ARGS[@]}" "${MISC_ARGS[@]}"
    "$@"
)

if [ "${DRY_RUN:-0}" != 1 ] && [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
runtime.setdefault("env_vars", {}).update({
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_CUMEM_ENABLE": "1",
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
    "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared",
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_DISABLE_MULTIMEM_AG": "1",
    "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
    "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
    "SGLANG_HEALTH_CHECK_TIMEOUT": "900",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
})
for name in ("CUDA_HOME", "GLOO_SOCKET_IFNAME", "CLEARML_CONFIG_FILE"):
    if os.environ.get(name):
        runtime["env_vars"][name] = os.environ[name]
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n%s\n' "${RUNTIME_ENV_JSON}"
    exit 0
fi
RAY_JOB_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.submit.log"
