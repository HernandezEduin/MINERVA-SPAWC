# MINERVA-InfoCost Experiment Results — 2026-08-28

This is the canonical tracker for the completed no-retraining rate-sweep campaign. Values come from the listed `rate_sweep_summary.csv`, `rate_sweep_summary.json`, and `rate_sweep_metadata.json` artifacts. All 22 directories were revalidated against their intended protocol and recorded checkpoint identity on 2026-08-28.

## 1. Scope and metric definitions

| Protocol | Operational unit | Utility | Communication paired with utility |
|---|---|---|---|
| `R=1` | One executed trajectory | Hits@1 equals trajectory success | Cost of that one path over all `T` hops |
| `R=100` | A 100-rollout sampled candidate ensemble | MINERVA `pool="max"` Hits@1/MRR | Total cost per question over all `100 × T` decisions |

`R=100` MRR is a sampled-rollout candidate-ranking metric: trajectories are scored, duplicate incorrect terminal entities are collapsed, and the first correct sampled terminal entity determines reciprocal rank. It is not conventional full-KG entity-ranking MRR.

Communication quantities remain distinct:

- **Fixed-rank bits:** hard local-rank payload under the retained support; not a source code.
- **Surprisal:** `-log2 p(a)` for the sampled execution-policy action; an information quantity, not automatically a realized packet length.
- **Shannon integer bits:** integer code-length diagnostic for sampled actions.
- **Entropy:** expected policy uncertainty, not an actual packet length.
- **Task-agnostic cost:** coding diagnostic under the smoothed graph-structural relation prior, which uses neither answer labels nor selected evaluation actions.

Lower PED and RED are better. PED and relation edit distance are separate metrics. MQuAKE-ST Multi has no compatible entity-path PED; MetaQA has neither compatible PED nor RED.

## 2. Artifact directory index

| Stage | Dataset | R | Cap | Rate seed | Modes | Output directory |
|---|---|---:|---:|---:|---|---|
| Original full | Kinship | 100 | 100 | 42 | Greedy, K=1–128, TF | `saved_models/kinshiphinton/20260827_191947` |
| Original full | MQuAKE-ST Single | 100 | 200 | 42 | Greedy, K=1–128, TF | `saved_models/mquake_st/20260827_193842` |
| Original full | MQuAKE-ST Multi | 100 | 200 | 42 | Greedy, K=1–128, TF | `saved_models/mquake_st/20260827_194422` |
| Original full | MetaQA | 100 | 200 | 42 | Greedy, K=1–128, TF | `saved_models/metaqa/20260827_194812` |
| Stage 1 | Kinship | 1 | 100 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/kinshiphinton/20260828_015559` |
| Stage 1 | MQuAKE-ST Single | 1 | 200 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/mquake_st/20260828_020226` |
| Stage 1 | MQuAKE-ST Multi | 1 | 200 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/mquake_st/20260828_020333` |
| Stage 1 | MetaQA | 1 | 200 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/metaqa/20260828_020631` |
| Stage 2 | MQuAKE-ST Single | 1 | 512 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/mquake_st/20260828_021451` |
| Stage 2 | MQuAKE-ST Multi | 1 | 512 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/mquake_st/20260828_021656` |
| Stage 2 | MQuAKE-ST Single | 100 | 512 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/mquake_st/20260828_021815` |
| Stage 2 | MQuAKE-ST Multi | 100 | 512 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/mquake_st/20260828_022724` |
| Stage 3A | Kinship | 100 | 100 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/kinshiphinton/20260828_021222` |
| Stage 3A | Kinship | 100 | 100 | 43 | Greedy, K=2, K=4, NumPy | `saved_models/kinshiphinton/20260828_021248` |
| Stage 3A | Kinship | 100 | 100 | 44 | Greedy, K=2, K=4, NumPy | `saved_models/kinshiphinton/20260828_021313` |
| Stage 3A | Kinship | 100 | 100 | 45 | Greedy, K=2, K=4, NumPy | `saved_models/kinshiphinton/20260828_021339` |
| Stage 3A | Kinship | 100 | 100 | 46 | Greedy, K=2, K=4, NumPy | `saved_models/kinshiphinton/20260828_021405` |
| Stage 3B | MetaQA | 100 | 200 | 42 | Greedy, K=2, K=4, NumPy | `saved_models/metaqa/20260828_030323` |
| Stage 3B | MetaQA | 100 | 200 | 43 | Greedy, K=2, K=4, NumPy | `saved_models/metaqa/20260828_040427` |
| Stage 3B | MetaQA | 100 | 200 | 44 | Greedy, K=2, K=4, NumPy | `saved_models/metaqa/20260828_050525` |
| Stage 3B | MetaQA | 100 | 200 | 45 | Greedy, K=2, K=4, NumPy | `saved_models/metaqa/20260828_060604` |
| Stage 3B | MetaQA | 100 | 200 | 46 | Greedy, K=2, K=4, NumPy | `saved_models/metaqa/20260828_070659` |

All runs use `use_beam=False` and `pool="max"`. Configured rollout count is 100 throughout; Stage 1 and Stage 2 `R=1` runs use the explicit evaluation override. Stage 2 keeps configured cap 200 and overrides only the effective evaluation cap to 512.

## 3. Original `R=100` full rate sweeps

In the tables below, communication outside brackets is bits/hop; `[total/q]` is total bits/question across the complete ensemble (`R × T`). `Hπ` is original-policy entropy in bits/hop, and `TA surp` is task-agnostic surprisal in bits/hop. TF is the upstream TensorFlow categorical reference.

### Kinship — 101 questions, 3 hops, cap 100

| Mode | Fixed/hop [total/q] | Hits@1 | MRR | PED | RED | Rollout success | Hπ | Surp/hop [total/q] | Shannon/hop [total/q] | TA surp | Trunc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy | .000 [0.0] | .960396 | .960396 | 1.8614 | 1.6436 | .960396 | 1.0377 | .0000 [0.0] | .0000 [0.0] | 2.4394 | 0% |
| K=1 | .000 [0.0] | .960396 | .960396 | 1.8614 | 1.6436 | .960396 | 1.0377 | .0000 [0.0] | .0000 [0.0] | 2.4394 | 0% |
| K=2 | 1.000 [300.0] | .970297 | .975248 | 1.9307 | 1.6931 | .966337 | .9963 | .5356 [160.7] | 1.2156 [364.7] | 2.5481 | 0% |
| K=4 | 2.000 [600.0] | .970297 | .980198 | 1.9901 | 1.7228 | .965248 | 1.0469 | .9484 [284.5] | 1.5663 [469.9] | 2.6187 | 0% |
| K=8 | 2.668 [800.3] | .960396 | .975248 | 1.9901 | 1.7129 | .962871 | 1.0776 | 1.0786 [323.6] | 1.7127 [513.8] | 2.6614 | 0% |
| K=16 | 2.668 [800.3] | .960396 | .975248 | 1.9901 | 1.7129 | .962871 | 1.0776 | 1.0786 [323.6] | 1.7127 [513.8] | 2.6614 | 0% |
| K=32 | 2.668 [800.3] | .960396 | .975248 | 1.9901 | 1.7129 | .962871 | 1.0776 | 1.0786 [323.6] | 1.7127 [513.8] | 2.6614 | 0% |
| K=64 | 2.668 [800.3] | .960396 | .975248 | 1.9901 | 1.7129 | .962871 | 1.0776 | 1.0786 [323.6] | 1.7127 [513.8] | 2.6614 | 0% |
| K=128 | 2.668 [800.3] | .960396 | .975248 | 1.9901 | 1.7129 | .962871 | 1.0776 | 1.0786 [323.6] | 1.7127 [513.8] | 2.6614 | 0% |
| TF | 2.671 [801.3] | .960396 | .976073 | 1.9901 | 1.7129 | .965248 | 1.0746 | 1.0782 [323.5] | 1.6855 [505.6] | 2.6654 | 0% |

### MQuAKE-ST Single — 1,504 questions, 4 hops, cap 200

| Mode | Fixed/hop [total/q] | Hits@1 | MRR | PED | RED | Rollout success | Hπ | Surp/hop [total/q] | Shannon/hop [total/q] | TA surp | Trunc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy | .000 [0.0] | .886303 | .886303 | .9215 | .6602 | .886303 | .0065 | .0000 [0.0] | .0000 [0.0] | 4.4074 | 5.170% |
| K=1 | .000 [0.0] | .886303 | .886303 | .9215 | .6602 | .886303 | .0065 | .0000 [0.0] | .0000 [0.0] | 4.4074 | 5.170% |
| K=2 | 1.000 [399.8] | .886303 | .888298 | .9215 | .6602 | .885246 | .0066 | .0063 [2.5] | .1219 [48.7] | 4.4065 | 5.147% |
| K=4 | 1.998 [799.3] | .886303 | .888298 | .9215 | .6602 | .885246 | .0066 | .0066 [2.6] | .1221 [48.8] | 4.4064 | 5.147% |
| K=8 | 2.987 [1194.6] | .886303 | .888298 | .9215 | .6602 | .885246 | .0066 | .0066 [2.6] | .1221 [48.8] | 4.4064 | 5.147% |
| K=16 | 3.924 [1569.7] | .886303 | .888298 | .9215 | .6602 | .885246 | .0066 | .0066 [2.6] | .1221 [48.8] | 4.4064 | 5.147% |
| K=32 | 4.615 [1846.2] | .886303 | .888298 | .9215 | .6602 | .885246 | .0066 | .0066 [2.6] | .1221 [48.8] | 4.4064 | 5.147% |
| K=64 | 5.053 [2021.1] | .886303 | .888298 | .9215 | .6602 | .885246 | .0066 | .0066 [2.6] | .1221 [48.8] | 4.4064 | 5.147% |
| K=128 | 5.224 [2089.4] | .886303 | .888298 | .9215 | .6602 | .885246 | .0066 | .0066 [2.6] | .1221 [48.8] | 4.4064 | 5.147% |
| TF | 5.288 [2115.3] | .886303 | .888852 | .9215 | .6602 | .885339 | .0066 | .0068 [2.7] | .0578 [23.1] | 4.4064 | 5.147% |

### MQuAKE-ST Multi — 870 questions, 4 hops, cap 200

| Mode | Fixed/hop [total/q] | Hits@1 | MRR | PED | RED | Rollout success | Hπ | Surp/hop [total/q] | Shannon/hop [total/q] | TA surp | Trunc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy | .000 [0.0] | .856322 | .856322 | — | 1.2023 | .856322 | .0110 | .0000 [0.0] | .0000 [0.0] | 4.5727 | 18.391% |
| K=1 | .000 [0.0] | .856322 | .856322 | — | 1.2023 | .856322 | .0110 | .0000 [0.0] | .0000 [0.0] | 4.5727 | 18.391% |
| K=2 | .999 [399.7] | .856322 | .860536 | — | 1.2023 | .854793 | .0109 | .0107 [4.3] | .1742 [69.7] | 4.5743 | 18.377% |
| K=4 | 1.994 [797.6] | .856322 | .860536 | — | 1.2023 | .854805 | .0109 | .0112 [4.5] | .1747 [69.9] | 4.5742 | 18.377% |
| K=8 | 2.972 [1188.8] | .856322 | .860536 | — | 1.2023 | .854805 | .0109 | .0112 [4.5] | .1747 [69.9] | 4.5742 | 18.377% |
| K=16 | 3.868 [1547.2] | .856322 | .860536 | — | 1.2023 | .854805 | .0109 | .0112 [4.5] | .1747 [69.9] | 4.5742 | 18.377% |
| K=32 | 4.548 [1819.3] | .856322 | .860536 | — | 1.2023 | .854805 | .0109 | .0112 [4.5] | .1747 [69.9] | 4.5742 | 18.377% |
| K=64 | 4.993 [1997.3] | .856322 | .860536 | — | 1.2023 | .854805 | .0109 | .0112 [4.5] | .1747 [69.9] | 4.5742 | 18.377% |
| K=128 | 5.272 [2108.7] | .856322 | .860536 | — | 1.2023 | .854805 | .0109 | .0112 [4.5] | .1747 [69.9] | 4.5742 | 18.377% |
| TF | 5.505 [2202.1] | .856322 | .860153 | — | 1.2023 | .854897 | .0110 | .0105 [4.2] | .0784 [31.3] | 4.5751 | 18.388% |

### MetaQA — 39,093 questions, 3 hops, cap 200

| Mode | Fixed/hop [total/q] | Hits@1 | MRR | PED | RED | Rollout success | Hπ | Surp/hop [total/q] | Shannon/hop [total/q] | TA surp | Trunc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy | .000 [0.0] | .897910 | .897910 | — | — | .897910 | .4567 | .0000 [0.0] | .0000 [0.0] | 3.5069 | 1.917% |
| K=1 | .000 [0.0] | .897910 | .897910 | — | — | .897910 | .4567 | .0000 [0.0] | .0000 [0.0] | 3.5069 | 1.917% |
| K=2 | 1.000 [300.0] | .897782 | .928080 | — | — | .895679 | .4549 | .2244 [67.3] | 1.0011 [300.3] | 3.5043 | 1.917% |
| K=4 | 1.896 [568.8] | .897859 | .929972 | — | — | .895019 | .4543 | .3262 [97.9] | 1.1081 [332.4] | 3.5032 | 1.917% |
| K=8 | 2.683 [804.9] | .897757 | .930104 | — | — | .894835 | .4540 | .3790 [113.7] | 1.1605 [348.1] | 3.5027 | 1.917% |
| K=16 | 3.210 [963.1] | .897757 | .930133 | — | — | .894754 | .4539 | .4099 [123.0] | 1.1916 [357.5] | 3.5023 | 1.917% |
| K=32 | 3.404 [1021.1] | .897731 | .930122 | — | — | .894743 | .4539 | .4304 [129.1] | 1.2114 [363.4] | 3.5023 | 1.917% |
| K=64 | 3.469 [1040.8] | .897782 | .930141 | — | — | .894733 | .4539 | .4451 [133.5] | 1.2260 [367.8] | 3.5025 | 1.917% |
| K=128 | 3.498 [1049.3] | .897757 | .930128 | — | — | .894714 | .4539 | .4527 [135.8] | 1.2336 [370.1] | 3.5026 | 1.917% |
| TF | 3.519 [1055.7] | .897757 | .930447 | — | — | .894668 | .4540 | .4540 [136.2] | .9117 [273.5] | 3.5026 | 1.917% |

The full sweeps show saturation rather than a monotonic rate–accuracy curve. Greedy/K=1 preserves ensemble Hits@1 closely, while stochastic support primarily improves sampled-candidate MRR. MQuAKE policy entropy and sampled surprisal are especially small despite large fixed-rank supports.

## 4. `R=1` single-trajectory results

Hits@1 equals rollout success exactly in every row. Communication is bits per executed path. Counts in parentheses are correct questions.

| Dataset | Mode | Correct; Hits@1/success | PED / RED | Path fixed / surprisal / Shannon | Hπ/hop | Mean valid | Trunc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Kinship | Greedy | 97; .960396 | 1.8614 / 1.6436 | 0 / 0 / 0 | 1.0377 | 5.977 | 0% |
| Kinship | K=2 | 99; .980198 | 1.9406 / 1.7228 | 3.000 / 1.548 / 3.574 | .9807 | 5.927 | 0% |
| Kinship | K=4 | 97; .960396 | 2.0000 / 1.7525 | 6.000 / 2.712 / 4.624 | 1.0275 | 6.125 | 0% |
| Kinship | NumPy | 97; .960396 | 1.9604 / 1.7327 | 8.089 / 3.133 / 5.079 | 1.0769 | 6.248 | 0% |
| MQuAKE Single | Greedy | 1,333; .886303 | .9215 / .6602 | 0 / 0 / 0 | .00651 | 43.929 | 5.170% |
| MQuAKE Single | K=2 | 1,331; .884973 | .9255 / .6629 | 3.998 / .0268 / .4847 | .00657 | 43.875 | 5.120% |
| MQuAKE Single | K=4 | 1,331; .884973 | .9249 / .6622 | 7.993 / .0273 / .4854 | .00657 | 43.872 | 5.120% |
| MQuAKE Single | NumPy | 1,331; .884973 | .9249 / .6622 | 21.160 / .0273 / .4854 | .00657 | 43.872 | 5.120% |
| MQuAKE Multi | Greedy | 745; .856322 | — / 1.2023 | 0 / 0 / 0 | .01096 | 65.252 | 18.391% |
| MQuAKE Multi | K=2 | 743; .854023 | — / 1.2011 | 3.997 / .0300 / .6793 | .01095 | 65.211 | 18.362% |
| MQuAKE Multi | K=4 | 743; .854023 | — / 1.2011 | 7.975 / .0310 / .6805 | .01095 | 65.211 | 18.362% |
| MQuAKE Multi | NumPy | 743; .854023 | — / 1.2011 | 22.013 / .0310 / .6805 | .01095 | 65.211 | 18.362% |
| MetaQA | Greedy | 35,102; .897910 | — / — | 0 / 0 / 0 | .4567 | 15.439 | 1.917% |
| MetaQA | K=2 | 35,025; .895940 | — / — | 3.000 / .674 / 3.004 | .4549 | 15.392 | 1.918% |
| MetaQA | K=4 | 35,008; .895506 | — / — | 5.688 / .982 / 3.328 | .4544 | 15.380 | 1.920% |
| MetaQA | NumPy | 34,999; .895275 | — / — | 10.555 / 1.363 / 3.707 | .4539 | 15.361 | 1.920% |

Important deltas versus greedy:

- Kinship K=2 gains 2/101 correct questions (`+0.019802`) but worsens both PED and RED by `+0.07921`. K=4 and NumPy do not improve success.
- MQuAKE Single and Multi stochastic modes each lose two correct questions. Their tiny path surprisal confirms near-deterministic pretrained policies.
- MetaQA K=2/K=4/NumPy lose 77/94/103 correct questions. Stochastic execution slightly hurts one-trajectory answer success.

## 5. MQuAKE cap 200 versus cap 512

The maximum raw valid-action count is 479. Effective cap 512 therefore reduces observed truncation to zero, but it is an evaluation-time sensitivity analysis of checkpoints trained under cap 200.

### `R=1`

Values use `cap 200 → cap 512`; communication is fixed/surprisal/Shannon bits per path.

| Dataset/mode | Hits@1/success | Correct Δ | PED / RED | Path communication | Hπ/hop | Truncation |
|---|---:|---:|---:|---:|---:|---:|
| Single greedy | .886303 → .888963 | +4 | .9215/.6602 → .9089/.6503 | 0 → 0 | .00651 → .00650 | 5.170% → 0% |
| Single K=2 | .884973 → .887633 | +4 | .9255/.6629 → .9116/.6516 | 3.998/.0268/.4847 → 4.000/.0260/.4847 | .00657 → .00656 | 5.120% → 0% |
| Single K=4 | .884973 → .887633 | +4 | .9249/.6622 → .9109/.6509 | 7.993/.0273/.4854 → 7.997/.0266/.4854 | .00657 → .00656 | 5.120% → 0% |
| Single NumPy | .884973 → .887633 | +4 | .9249/.6622 → .9109/.6509 | 21.160/.0273/.4854 → 21.266/.0266/.4854 | .00657 → .00656 | 5.120% → 0% |
| Multi greedy | .856322 → .840230 | −14 | —/1.2023 → —/1.3356 | 0 → 0 | .01096 → .01719 | 18.391% → 0% |
| Multi K=2 | .854023 → .839080 | −13 | —/1.2011 → —/1.3333 | 3.997/.0300/.6793 → 3.997/.0513/1.0253 | .01095 → .01718 | 18.362% → 0% |
| Multi K=4 | .854023 → .839080 | −13 | —/1.2011 → —/1.3333 | 7.975/.0310/.6805 → 7.975/.0524/1.0264 | .01095 → .01718 | 18.362% → 0% |
| Multi NumPy | .854023 → .839080 | −13 | —/1.2011 → —/1.3333 | 22.013/.0310/.6805 → 22.598/.0524/1.0264 | .01095 → .01718 | 18.362% → 0% |

Single changes by only four questions (`+0.2660` percentage points). Multi is materially cap-sensitive: greedy loses 14 questions (`−1.6092` points) and RED worsens by `+0.1333`.

### `R=100`

Matched modes use `cap 200 → cap 512`; communication is total fixed/surprisal/Shannon bits per question across all 100 four-hop rollouts.

| Dataset/mode | Hits@1 | MRR | Rollout success | PED / RED | Hπ/hop | Total question communication | Truncation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single greedy | .886303 → .888963 | .886303 → .888963 | .886303 → .888963 | .9215/.6602 → .9089/.6503 | .00651 → .00650 | 0 → 0 | 5.170% → 0% |
| Single K=2 | .886303 → .888963 | .888298 → .891290 | .885246 → .887819 | .9215/.6602 → .9089/.6503 | .00661 → .00660 | 399.80/2.531/48.742 → 400.00/2.522/48.812 | 5.147% → 0% |
| Single K=4 | .886303 → .888963 | .888298 → .891290 | .885246 → .887819 | .9215/.6602 → .9089/.6503 | .00661 → .00660 | 799.27/2.623/48.830 → 799.67/2.614/48.900 | 5.147% → 0% |
| Multi greedy | .856322 → .840230 | .856322 → .840230 | .856322 → .840230 | —/1.2023 → —/1.3356 | .01096 → .01719 | 0 → 0 | 18.391% → 0% |
| Multi K=2 | .856322 → .840230 | .860536 → .848467 | .854793 → .839966 | —/1.2023 → —/1.3356 | .01094 → .01716 | 399.74/4.284/69.672 → 399.74/6.690/104.446 | 18.377% → 0% |
| Multi K=4 | .856322 → .840230 | .860536 → .848467 | .854805 → .839943 | —/1.2023 → —/1.3356 | .01094 → .01716 | 797.64/4.481/69.883 → 797.64/6.940/104.689 | 18.377% → 0% |

The original cap-200 runs predate explicit `mean_question_*` fields; their totals above are reconstructed exactly from their per-hop/per-path fields and `R=100`.

| Dataset/cap/reference | Hits@1 | MRR | Rollout success | Total fixed / surprisal / Shannon |
|---|---:|---:|---:|---:|
| Single, cap 200 TF | .886303 | .888852 | .885339 | 2115.30 / 2.706 / 23.112 |
| Single, cap 512 NumPy | .888963 | .891290 | .887819 | 2125.94 / 2.614 / 48.900 |
| Multi, cap 200 TF | .856322 | .860153 | .854897 | 2202.12 / 4.207 / 31.349 |
| Multi, cap 512 NumPy | .840230 | .848467 | .839943 | 2260.14 / 6.940 / 104.689 |

TF and NumPy use different RNG backends, so these unrestricted rows are references rather than a clean cap-only pair. The matched K=2/K=4 rows establish the conclusion: removing truncation changes the common baseline, especially for Multi, but does not reveal a stochastic-navigation advantage. MQuAKE remains near-deterministic and rate-flat within each cap.

## 6. Kinship five-seed `R=100`

Population standard deviation across rate seeds 42–46 is used. PED/RED are metrics for the highest-ranked selected trajectory; communication is total bits/question across all 100 three-hop rollouts.

| Mode | Hits@1 mean ± SD [range] | MRR mean ± SD [range] | PED | RED | Rollout success mean ± SD |
|---|---:|---:|---:|---:|---:|
| Greedy | .960396 ± 0 [.960396,.960396] | .960396 ± 0 [.960396,.960396] | 1.861386 ± 0 | 1.643564 ± 0 | .960396 ± 0 |
| K=2 | .968317 ± .003960 [.960396,.970297] | .974257 ± .001980 [.970297,.975248] | 1.930693 ± 0 | 1.693069 ± 0 | .966871 ± .000924 |
| K=4 | .968317 ± .003960 [.960396,.970297] | .980528 ± .003025 [.975248,.983498] | 1.990099 ± 0 | 1.722772 ± 0 | .967188 ± .001121 |
| NumPy | .960396 ± 0 [.960396,.960396] | .975908 ± .000962 [.975248,.977723] | 1.990099 ± 0 | 1.712871 ± 0 | .965307 ± .001347 |

| Mode | Total fixed bits/question | Total surprisal bits/question | Total Shannon bits/question |
|---|---:|---:|---:|
| Greedy | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| K=2 | 300.00 ± 0 | 160.91 ± .49 | 364.91 ± .42 |
| K=4 | 600.00 ± 0 | 284.17 ± .45 | 469.61 ± .64 |
| NumPy | 801.11 ± .55 | 323.35 ± .40 | 513.38 ± .57 |

### Paired seed deltas

| Comparison | Δ Hits@1 mean ± SD [range] | Δ MRR mean ± SD [range] |
|---|---:|---:|
| K=2 − greedy | +.007921 ± .003960 [0,+.009901] | +.013861 ± .001980 [.009901,.014851] |
| K=4 − greedy | +.007921 ± .003960 [0,+.009901] | +.020132 ± .003025 [.014851,.023102] |
| NumPy − greedy | 0 ± 0 | +.015512 ± .000962 [.014851,.017327] |
| K=2 − NumPy | +.007921 ± .003960 [0,+.009901] | −.001650 ± .001881 [−.004950,0] |
| K=4 − NumPy | +.007921 ± .003960 [0,+.009901] | +.004620 ± .002481 [0,.007426] |

One Kinship question is `1/101 = 0.009901` Hits@1. K=2/K=4 gain exactly one question in seeds 42, 43, 45, and 46 and tie greedy in seed 44. NumPy Hits@1 equals greedy in all five seeds. This recurring one-question outcome is not sufficient to claim established inference regularization. Stochastic modes improve ensemble MRR but have worse PED/RED and nonzero communication.

## 7. MetaQA five-seed `R=100`

Population standard deviation across rate seeds 42–46 is used. Communication is total bits/question across all 100 three-hop rollouts.

| Mode | Hits@1 mean ± SD [range] | Δ questions vs greedy | MRR mean ± SD [range] | Rollout success mean ± SD |
|---|---:|---:|---:|---:|
| Greedy | .897910 ± 0 [.897910,.897910] | 0 | .897910 ± 0 [.897910,.897910] | .897910 ± 0 |
| K=2 | .897782 ± 0 [.897782,.897782] | −5 every seed | .928047 ± .000038 [.928010,.928104] | .895606 ± .000070 |
| K=4 | .897859 ± 0 [.897859,.897859] | −2 every seed | .930113 ± .000083 [.929972,.930217] | .895035 ± .000068 |
| NumPy | .897792 ± .000013 [.897782,.897808] | mean −4.6 [−5,−4] | .930280 ± .000088 [.930141,.930385] | .894724 ± .000068 |

| Mode | Fixed bits/question | Surprisal bits/question | Shannon bits/question | Entropy-sum bits/question | Truncation |
|---|---:|---:|---:|---:|---:|
| Greedy | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 1.9168% ± 0 |
| K=2 | 300.000 ± 0 | 67.330 ± .037 | 300.356 ± .039 | 67.358 ± .002 | 1.9162% ± .0003% |
| K=4 | 568.817 ± .003 | 97.898 ± .045 | 332.445 ± .044 | 97.940 ± .012 | 1.9169% ± .0003% |
| NumPy | 1055.663 ± .012 | 136.122 ± .069 | 370.380 ± .068 | 136.182 ± .014 | 1.9171% ± .0003% |

Mean valid-action counts are 15.4392/15.3975/15.3734/15.3620 for greedy/K=2/K=4/NumPy. Task-agnostic totals are approximately 1051–1052 surprisal and 1211–1213 Shannon bits/question, substantially above the task-conditioned sampled surprisal.

### Paired seed deltas

| Comparison | Δ Hits@1 mean ± SD [range] | Approx. question difference | Δ MRR mean ± SD [range] |
|---|---:|---:|---:|
| K=2 − greedy | −.000128 ± 0 | −5 every seed | +.030137 ± .000038 [.030100,.030194] |
| K=4 − greedy | −.000051 ± 0 | −2 every seed | +.032203 ± .000083 [.032062,.032307] |
| NumPy − greedy | −.000118 ± .000013 | mean −4.6 [−5,−4] | +.032370 ± .000088 [.032231,.032474] |
| K=2 − NumPy | −.000010 ± .000013 | mean −0.4 [−1,0] | −.002233 ± .000117 [−.002375,−.002061] |
| K=4 − NumPy | +.000067 ± .000013 | mean +2.6 [+2,+3] | −.000167 ± .000020 [−.000199,−.000135] |

### Backend-matched MRR recovery

For each seed, `G = MRR_NumPy − MRR_greedy` and `recovery_K = (MRR_K − MRR_greedy)/G`.

| Seed | NumPy gain G | K=2 recovery | K=4 recovery |
|---:|---:|---:|---:|
| 42 | .032231 | 93.606% | 99.476% |
| 43 | .032340 | 93.363% | 99.498% |
| 44 | .032452 | 92.794% | 99.387% |
| 45 | .032474 | 92.687% | 99.485% |
| 46 | .032350 | 93.068% | 99.582% |
| **Mean ± population SD** | — | **93.103% ± .344%** | **99.486% ± .062%** |

K=2 recovers most of the unrestricted NumPy MRR benefit. K=4 adds `0.002066 ± 0.000113` MRR beyond K=2 and nearly saturates the NumPy gain, but requires about 269 additional fixed-rank bits/question. The large MRR gain and near-zero Hits@1 change are stable across all five rate seeds.

## 8. `R=1` versus `R=100`

MetaQA most clearly separates trajectory utility from ensemble ranking utility.

| Mode | `R=1` trajectory success | `R=100` per-rollout success | `R=100` ensemble Hits@1 | `R=100` ensemble MRR | `R=100` total fixed bits/question |
|---|---:|---:|---:|---:|---:|
| Greedy | .897910 | .897910 | .897910 | .897910 | 0 |
| K=2 | .895940 | .895606 ± .000070 | .897782 | .928047 | 300.000 |
| K=4 | .895506 | .895035 ± .000068 | .897859 | .930113 | 568.817 |
| NumPy | .895275 | .894724 ± .000068 | .897792 ± .000013 | .930280 | 1055.663 |

Stochasticity slightly hurts one executed trajectory and average per-rollout success. With 100 sampled trajectories, ensemble Hits@1 nearly recovers to greedy while MRR improves by about 0.030–0.032. That ranking gain must be paired with the total ensemble communication above, not only a nominal per-hop rank.

## 9. Cross-dataset conclusions

Evidence directly supported by the completed artifacts:

1. `R=1` and `R=100` are different communication tasks with different utility units.
2. Stochastic `R=1` execution has no general answer-success advantage. Kinship K=2 is the exception, gaining two questions at seed 42 while worsening PED/RED; MQuAKE and MetaQA stochastic modes underperform greedy slightly.
3. Under `R=100`, deterministic greedy preserves essentially all top-1 utility, while stochastic branching improves sampled-candidate MRR on Kinship and MetaQA.
4. MetaQA is the cleanest ranked-output example: stochastic modes differ from greedy by only 2–5 Hits@1 questions among 39,093 but gain roughly 0.030–0.032 MRR with very small seed variation.
5. Kinship provides complementary path evidence: stochastic MRR improves while selected-path PED/RED worsen.
6. MQuAKE remains near-deterministic. Removing cap-200 truncation changes absolute performance, particularly Multi, but does not expose a stochastic-navigation advantage.
7. The graph-structural task-agnostic prior is substantially more expensive than the task-conditioned execution policy.
8. There is no monotonic communication-versus-accuracy law in these experiments.

Safe summary:

> Under these pretrained checkpoints and evaluation protocols, deterministic shared-policy execution preserves essentially all top-1 utility, while stochastic branching improves the quality of the sampled ranked-candidate set in the `R=100` protocol. The Kinship `R=1`, K=2 two-question improvement is a documented exception to any blanket greedy-best claim.

## 10. Scientific caveats

- No retraining was performed; every result uses a fixed pretrained policy.
- Zero greedy action payload assumes synchronized policy, task/state, candidate action set, action ordering, and tie-breaking. It is not a universal zero-communication theorem and excludes the cost of establishing that shared side information.
- `R=100` Hits@1/MRR are MINERVA sampled-candidate ranking metrics under `pool="max"`, not conventional full-KG entity-ranking metrics.
- Any `R=100` ranking claim must report total ensemble communication across all rollouts and hops.
- Entropy is an expectation, not an actual packet length.
- Sampled surprisal is not automatically a realized code length.
- Shannon integer length, sampled surprisal, entropy, and fixed-rank support cost are distinct quantities.
- Top-K fixed-rank cost is a retained-support/local-rank bound, not source coding.
- MetaQA remains action-truncated at cap 200: maximum raw action count is 4,305 and visited-state truncation is about 1.917%.
- MQuAKE cap 512 is an evaluation-only sensitivity analysis. The checkpoints were trained with cap 200.
- Common PCG64 initialization gives matched random draws only while Top-K and NumPy states remain comparable; trajectories are no longer paired after their states diverge.
- Candidate diversity is a plausible explanation for the stochastic MRR gain, but no explicit unique-candidate diversity metric was measured.
- The task-agnostic prior is graph-structural and smoothed; it is not fit from evaluation labels or selected test actions.
- Five rate seeds quantify sampling stability but do not alone establish statistical significance.
- PED and RED are not interchangeable; unavailable metrics remain marked `—` rather than zero.

## 11. Final experimental verdict

- **Core experimental status:** COMPLETE
- **Additional core experiments required:** NO
- **Next phase:** ICASSP manuscript revision
