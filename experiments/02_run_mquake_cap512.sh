#!/usr/bin/env bash
set -euo pipefail

# MINERVA-InfoCost
# Stage 2: MQuAKE action-cap sensitivity at max_num_actions=512
# CPU-only. Uses Conda environment: minerva_tf2
#
# Runs both R=1 and R=100 for MQuAKE-ST Single and Multi.
#
# Run from the repository root:
#   bash 02_run_mquake_cap512.sh

echo "=== Stage 2: MQuAKE cap=512 sensitivity ==="

echo
echo ">>> MQuAKE-ST Single | R=1 | cap=512"
conda run -n minerva_tf2 bash run_rate_sweep.sh configs/mquake_st_single.yaml \
  --rate_test_rollouts 1 \
  --rate_max_num_actions_override 512 \
  --rate_top_k 2 4 \
  --rate_include_numpy_policy true \
  --rate_include_unrestricted false \
  --rate_seed 42

echo
echo ">>> MQuAKE-ST Multi | R=1 | cap=512"
conda run -n minerva_tf2 bash run_rate_sweep.sh configs/mquake_st_multi.yaml \
  --rate_test_rollouts 1 \
  --rate_max_num_actions_override 512 \
  --rate_top_k 2 4 \
  --rate_include_numpy_policy true \
  --rate_include_unrestricted false \
  --rate_seed 42

echo
echo ">>> MQuAKE-ST Single | R=100 | cap=512"
conda run -n minerva_tf2 bash run_rate_sweep.sh configs/mquake_st_single.yaml \
  --rate_test_rollouts 100 \
  --rate_max_num_actions_override 512 \
  --rate_top_k 2 4 \
  --rate_include_numpy_policy true \
  --rate_include_unrestricted false \
  --rate_seed 42

echo
echo ">>> MQuAKE-ST Multi | R=100 | cap=512"
conda run -n minerva_tf2 bash run_rate_sweep.sh configs/mquake_st_multi.yaml \
  --rate_test_rollouts 100 \
  --rate_max_num_actions_override 512 \
  --rate_top_k 2 4 \
  --rate_include_numpy_policy true \
  --rate_include_unrestricted false \
  --rate_seed 42

echo
echo "=== Stage 2 complete ==="
