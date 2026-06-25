#!/usr/bin/env bash
# Entrypoint for the ZyinkLabs training container.
#
# RUN_MODE selects which pipeline stage this container invocation performs.
# Set via docker-compose.yml per service.
set -euo pipefail

RUN_MODE="${RUN_MODE:-train}"

echo "== ZyinkLabs container starting (RUN_MODE=${RUN_MODE}) =="

case "$RUN_MODE" in

  download_computage_bench)
    # Downloads the public ComputAgeBench dataset (training + benchmark splits).
    # This replaces the wet-lab IDAT processing stage for Version 1, which uses
    # only public data -- see docs/DATASET.md for why ComputAgeBench was chosen.
    echo "--> Downloading ComputAgeBench dataset..."
    python /workspace/scripts/01_download_computage_bench.py \
      --out_dir "/workspace/data/processed/computage"
    echo "== ComputAgeBench download complete =="
    ;;

  prepare_data)
    # Converts the downloaded ComputAgeBench training split into CpGPT's
    # memory-mapped format, ready for cpgpt-train. Run download_computage_bench
    # first.
    RIDGE_ARGS=()
    if [ -n "${RIDGE_SELECT_TOP_N:-}" ]; then
      RIDGE_ARGS=(--ridge_select_top_n "${RIDGE_SELECT_TOP_N}")
      echo "--> Ridge feature selection enabled: top ${RIDGE_SELECT_TOP_N} CpG sites"
    fi

    echo "--> Preparing CpGPT memory-mapped data for target: ${TARGET_COLUMN:?TARGET_COLUMN must be set}"
    python /workspace/scripts/02_prepare_cpgpt_data.py \
      --input "${PREPARE_INPUT:-/workspace/data/processed/computage/computage_train_merged.feather}" \
      --target "${TARGET_COLUMN}" \
      --out_dir "/workspace/CpGPT/data/zyinklabs/cpgpt/${EXPERIMENT_NAME:?EXPERIMENT_NAME must be set}" \
      --cpgpt_dir "/workspace/CpGPT" \
      "${RIDGE_ARGS[@]}"

    echo "== Data preparation complete =="
    ;;

  download_dependencies)
    # One-time pull of CpGPT's pretrained weights + DNA LLM embeddings from S3.
    # Requires AWS credentials to be available in the container (see compose file).
    python -c "
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
inferencer = CpGPTInferencer(dependencies_dir='/workspace/CpGPT/dependencies', data_dir='/workspace/CpGPT/data')
inferencer.download_dependencies(species='human')
inferencer.download_model('${MODEL_NAME:-large}')
"
    echo "== Dependency download complete =="
    ;;

  train)
    # Stage: finetune + validate + conditionally push to Hugging Face.
    # This is the GPU-bound stage -- see orchestrate_training.py for the gating logic.
    python /workspace/orchestrate_training.py
    ;;

  benchmark)
    # Compares our finetuned model against SOTA published clocks (Horvath, Hannum,
    # PhenoAge, etc.) using the official ComputAgeBench methodology, on the
    # held-out benchmark split. See scripts/04_benchmark_vs_sota.py.
    python /workspace/scripts/04_benchmark_vs_sota.py \
      --checkpoint "${CHECKPOINT_PATH:?CHECKPOINT_PATH must be set}" \
      --config "${CONFIG_PATH:?CONFIG_PATH must be set}" \
      --benchmark_data "${BENCHMARK_DATA:-/workspace/data/processed/computage/computage_benchmark_merged.feather}" \
      --cpgpt_dir "/workspace/CpGPT" \
      --out_dir "${BENCHMARK_OUT_DIR:-/workspace/data/processed/benchmark_results}"
    ;;

  inference)
    python /workspace/scripts/03_run_inference.py \
      --checkpoint "${CHECKPOINT_PATH:?CHECKPOINT_PATH must be set}" \
      --config "${CONFIG_PATH:?CONFIG_PATH must be set}" \
      --input "${INFERENCE_INPUT:?INFERENCE_INPUT must be set}" \
      --out "${INFERENCE_OUTPUT:-/workspace/data/processed/predictions.csv}" \
      --cpgpt_dir "/workspace/CpGPT"
    ;;

  shell)
    # Drop into a shell for debugging -- useful with `docker compose run`.
    exec /bin/bash
    ;;

  *)
    echo "Unknown RUN_MODE: ${RUN_MODE}"
    echo "Valid options: download_computage_bench | download_dependencies | prepare_data | train | benchmark | inference | shell"
    exit 1
    ;;
esac
