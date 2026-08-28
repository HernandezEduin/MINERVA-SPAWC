#!/usr/bin/env bash
set -euo pipefail

# Post-review primary matrix: deterministic diversity versus stochastic execution.
# CPU-only. Uses the fixed pretrained checkpoints through minerva_tf2.
#
# Run from the repository root:
#   bash experiments/05_run_r100_diversity_all_datasets.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=""

configs=(
  configs/kinshiphinton.yaml
  configs/metaqa.yaml
  configs/mquake_st_single.yaml
  configs/mquake_st_multi.yaml
)
labels=(
  "Kinship | cap=100"
  "MetaQA | cap=200"
  "MQuAKE-ST Single | cap=200"
  "MQuAKE-ST Multi | cap=200"
)

echo "=== Post-review primary matrix: R=100, seed=42 ==="
echo "Modes: greedy, Top-2, Top-4, NumPy unrestricted, deterministic beam-100"

for index in "${!configs[@]}"; do
  echo
  echo ">>> ${labels[$index]} | R=100 | seed=42"
  conda run -n minerva_tf2 bash run_rate_sweep.sh "${configs[$index]}" \
    --rate_test_rollouts 100 \
    --rate_top_k 2 4 \
    --rate_include_numpy_policy true \
    --rate_include_deterministic_beam true \
    --rate_beam_width 100 \
    --rate_include_unrestricted false \
    --rate_seed 42
done

echo
echo "=== Post-review primary matrix complete ==="
