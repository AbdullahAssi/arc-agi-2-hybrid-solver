#!/usr/bin/env bash
# Run llm.infer with vLLM on the GPU server.
# FlashInfer's sampling kernel is compiled on first use, and that compile fails here (the
# system's CUDA 11.5 headers clash with CUDA 13). Greedy decoding doesn't need it, so use
# vLLM's prebuilt sampler. ninja and a CUDA 13 nvcc are still put first on PATH for any
# other kernels that get compiled.
# Usage: bash run_vllm.sh --split evaluation --augs 8 --tag NAME [--adapter DIR]
set -euo pipefail
cd "$(dirname "$0")"
VENV=$PWD/.venv-vllm
export CUDA_HOME=$VENV/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$VENV/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
exec python -m llm.infer --engine vllm "$@"
