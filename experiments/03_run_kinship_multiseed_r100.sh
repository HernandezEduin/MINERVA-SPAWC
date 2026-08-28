#!/usr/bin/env bash
set -euo pipefail

# MINERVA-InfoCost
# Stage 3A: Kinship R=100, five evaluation seeds
# CPU-only. Uses Conda environment: minerva_tf2
#
# Modes produced:
#   greedy (always included by evaluator)
#   Top-K K=2
#   Top-K K=4
#   numpy_policy
#
# Seeds: 42, 43, 44, 45, 46
#
# Run from the repository root:
#   bash 03_run_kinship_multiseed_r100.sh

echo "=== Stage 3A: Kinship R=100 multi-seed ==="

for s in 42 43 44 45 46; do
  echo
  echo ">>> Kinship | R=100 | seed=${s}"
  conda run -n minerva_tf2 bash run_rate_sweep.sh configs/kinshiphinton.yaml \
    --rate_test_rollouts 100 \
    --rate_top_k 2 4 \
    --rate_include_numpy_policy true \
    --rate_include_unrestricted false \
    --rate_seed "$s"
done

echo
echo "=== Stage 3A complete ==="
