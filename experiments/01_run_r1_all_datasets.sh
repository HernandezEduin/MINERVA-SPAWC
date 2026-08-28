#!/usr/bin/env bash
set -euo pipefail

# MINERVA-InfoCost
# Stage 1: Full-dataset single-trajectory evaluation (R=1)
# CPU-only. Uses Conda environment: minerva_tf2
#
# Run from the repository root:
#   bash 01_run_r1_all_datasets.sh

echo "=== Stage 1: R=1 single-trajectory evaluation ==="

echo
echo ">>> Kinship"
conda run -n minerva_tf2 bash run_rate_sweep.sh configs/kinshiphinton.yaml \
  --rate_test_rollouts 1 \
  --rate_top_k 2 4 \
  --rate_include_numpy_policy true \
  --rate_include_unrestricted false \
  --rate_seed 42

echo
echo ">>> MQuAKE-ST Single"
conda run -n minerva_tf2 bash run_rate_sweep.sh configs/mquake_st_single.yaml \
  --rate_test_rollouts 1 \
  --rate_top_k 2 4 \
  --rate_include_numpy_policy true \
  --rate_include_unrestricted false \
  --rate_seed 42

echo
echo ">>> MQuAKE-ST Multi"
conda run -n minerva_tf2 bash run_rate_sweep.sh configs/mquake_st_multi.yaml \
  --rate_test_rollouts 1 \
  --rate_top_k 2 4 \
  --rate_include_numpy_policy true \
  --rate_include_unrestricted false \
  --rate_seed 42

echo
echo ">>> MetaQA"
conda run -n minerva_tf2 bash run_rate_sweep.sh configs/metaqa.yaml \
  --rate_test_rollouts 1 \
  --rate_top_k 2 4 \
  --rate_include_numpy_policy true \
  --rate_include_unrestricted false \
  --rate_seed 42

echo
echo "=== Stage 1 complete ==="
