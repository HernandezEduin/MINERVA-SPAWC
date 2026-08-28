#!/usr/bin/env bash
set -euo pipefail

# Evaluation-only action-cap sensitivity. The checkpoints were trained at cap=200;
# cap=512 does not represent retraining or a full-action training condition.
# CPU-only. Uses Conda environment: minerva_tf2.
#
# Run from the repository root:
#   bash experiments/06_run_mquake_beam_cap512.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=""

configs=(
  configs/mquake_st_single.yaml
  configs/mquake_st_multi.yaml
)
labels=(
  "MQuAKE-ST Single"
  "MQuAKE-ST Multi"
)

echo "=== MQuAKE deterministic-beam cap=512 sensitivity ==="
echo "Modes: greedy, Top-2, Top-4, NumPy unrestricted, deterministic beam-100"

for index in "${!configs[@]}"; do
  echo
  echo ">>> ${labels[$index]} | R=100 | beam=100 | evaluation cap=512"
  conda run -n minerva_tf2 bash run_rate_sweep.sh "${configs[$index]}" \
    --rate_test_rollouts 100 \
    --rate_max_num_actions_override 512 \
    --rate_top_k 2 4 \
    --rate_include_numpy_policy true \
    --rate_include_deterministic_beam true \
    --rate_beam_width 100 \
    --rate_include_unrestricted false \
    --rate_seed 42
done

echo
echo "=== MQuAKE cap=512 sensitivity complete ==="
