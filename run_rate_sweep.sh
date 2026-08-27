#!/bin/bash

config="$1"
shift
export PYTHONPATH=".:./minerva:${PYTHONPATH:-}"
gpu_id=""
if [[ $# -gt 0 && "$1" != --* ]]; then
  gpu_id="$1"
  shift
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

cmd=(python code/evaluation_rate_sweep.py --config_yaml "$config" "$@")

if [[ -z "$gpu_id" ]] || ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -i "$gpu_id" >/dev/null 2>&1; then
  echo "Executing (CPU): ${cmd[*]}"
  CUDA_VISIBLE_DEVICES="" "${cmd[@]}"
else
  echo "Executing: CUDA_VISIBLE_DEVICES=$gpu_id ${cmd[*]}"
  CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}"
fi
