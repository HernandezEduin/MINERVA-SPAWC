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

## 12. Post-review deterministic-diversity experiments pending — 2026-08-29

The Section 11 verdict is retained as the historical conclusion of the 2026-08-28 campaign, but its statement that no additional core experiment was required is **superseded** by an independent manuscript/reviewer concern: multiple candidates may be generated deterministically by synchronized beam decoding with zero incremental stochastic action-realization payload.

Evaluation support and CPU launchers are implemented at root HEAD `5a1d29aeac6cfde3ad55f6b723000db433511812`, with MINERVA still pinned at `9bf1ae998d14471c3f7c31f70969d0bbf9873329`. The planned seed-42 `R=100` comparison is greedy / Top-2 / Top-4 / NumPy unrestricted / deterministic beam-100 on Kinship, MetaQA, MQuAKE-ST Single, and MQuAKE-ST Multi, followed by the evaluation-only MQuAKE cap-512 sensitivity.

No full post-review result is recorded yet. A one-batch Kinship validation artifact exists at `saved_models/kinshiphinton/20260829_014259/rate_sweep`; its external beam Hits@1/MRR (`0.96875` / `0.98046875`) matches pinned upstream beam exactly, but this smoke result must not be used as a full-dataset scientific conclusion. After the user runs `experiments/05_run_r100_diversity_all_datasets.sh` and `experiments/06_run_mquake_beam_cap512.sh`, append exact artifact paths, metrics, diversity/coverage decomposition, cap sensitivity, and the Case A/B/C interpretation here.

## 13. Post-review deterministic-diversity experiments — 2026-08-29

This section resolves the pending experiment recorded in Section 12. It is self-contained and does not replace the historical results above.

### 13.1 Motivation

The experiment tests whether candidate diversity requires communicated stochastic action realization, or whether synchronized endpoints can obtain comparable ranking utility by running the same deterministic beam decoder. The tested beam uses the fixed pretrained policy, cumulative policy log probability, 100 retained beam slots, and the pinned MINERVA `pool="max"` ranking semantics. Under identical policy/state/local-action-interface/order/tie-breaking, its incremental stochastic action-realization payload is zero. This is not a claim of zero total communication or zero computation.

### 13.2 Artifact index and protocol integrity

| Campaign | Dataset | Questions | Configured → effective cap | Output directory |
|---|---|---:|---:|---|
| Primary | Kinship | 101 | 100 → 100 | `saved_models/kinshiphinton/20260829_030334` |
| Primary | MetaQA | 39,093 | 200 → 200 | `saved_models/metaqa/20260829_030347` |
| Primary | MQuAKE-ST Single / SA | 1,504 | 200 → 200 | `saved_models/mquake_st/20260829_042015` |
| Primary | MQuAKE-ST Multi / MA | 870 | 200 → 200 | `saved_models/mquake_st/20260829_042415` |
| Cap sensitivity | MQuAKE-ST Single / SA | 1,504 | 200 → 512 | `saved_models/mquake_st/20260829_030706` |
| Cap sensitivity | MQuAKE-ST Multi / MA | 870 | 200 → 512 | `saved_models/mquake_st/20260829_031635` |

All six CSV/JSON/metadata triplets were present and mutually consistent. Every run records root HEAD `6120683e8a0713db890f0b8b301a7dd9e034906a`, MINERVA SHA `9bf1ae998d14471c3f7c31f70969d0bbf9873329`, `R=100`, rate seed 42, the full question count, `pool="max"`, and the intended checkpoint. Checkpoint restore compatibility and unchanged checkpoint identity are true in every metadata file. All primary rows contain Greedy, Top-2, Top-4, NumPy unrestricted, and deterministic beam in that order. Beam requested/effective width is 100; every row has 100 logical candidate slots.

The recorded checkpoint prefixes are `saved_models/kinshiphinton/qa_nhop_reason_3hop_seed42/model/model.ckpt`, `saved_models/metaqa/qa_nhop_reason_3hop_seed42/model/model.ckpt`, `saved_models/mquake_st/sa_qa_nhop_reason_4hop_seed42/model/model.ckpt`, and `saved_models/mquake_st/ma_qa_nhop_reason_4hop_seed42/model/model.ckpt`; recorded sizes match the current checkpoint files. Sampling backends are deterministic argmax for greedy, seeded NumPy PCG64 for Top-K/NumPy unrestricted, and the deterministic pinned-MINERVA beam mirror with default NumPy `argsort` for beam.

The cap-512 runs use the same cap-200-trained SA/MA checkpoints, with configured cap 200 and evaluation-only effective cap 512. Maximum raw valid degree remains 479 and observed truncation becomes zero. These are sensitivity results, not retrained or full-action-trained models.

Numerical checks passed: greedy MRR equals Hits@1; greedy terminal diversity is exactly one; greedy coverage equals Hits@1 and rollout success; coverage is never below Hits@1; candidate fractions are bounded; no unique count exceeds 100; Top-2 payload is approximately 300 bits/question for three-hop datasets and 400 for four-hop datasets; beam stochastic payload is zero; and beam fixed-rank, surprisal, Shannon, and execution-policy coding fields remain null. For beam, `rollout_success_rate` is the fraction of retained beam slots ending at an answer, not the success probability of a randomly sampled rollout.

### 13.3 Compact cross-dataset scientific comparison

All communication values in this table are total stochastic action-realization payload per question over the 100-slot, full-horizon protocol. NumPy is the full-support local-rank reference. Beam has no fixed-rank branch-message value; its displayed zero is only the explicitly emitted stochastic-realization payload.

| Dataset | Mode | Hits@1 | MRR | Candidate coverage | Mean unique terminals | Stochastic payload/q |
|---|---|---:|---:|---:|---:|---:|
| Kinship | Greedy | 0.960396 | 0.960396 | 0.960396 | 1.000 | 0.000 |
| Kinship | Top-2 | 0.970297 | 0.975248 | 0.980198 | 1.158 | 300.000 |
| Kinship | Top-4 | 0.970297 | 0.980198 | 0.990099 | 1.257 | 600.000 |
| Kinship | NumPy unrestricted | 0.960396 | 0.975248 | 0.990099 | 1.327 | 800.287 |
| Kinship | Beam-100 | 0.960396 | 0.977723 | 1.000000 | 9.109 | 0.000 |
| MetaQA | Greedy | 0.897910 | 0.897910 | 0.897910 | 1.000 | 0.000 |
| MetaQA | Top-2 | 0.897782 | 0.928080 | 0.960965 | 1.849 | 300.000 |
| MetaQA | Top-4 | 0.897859 | 0.929972 | 0.966234 | 2.562 | 568.814 |
| MetaQA | NumPy unrestricted | 0.897782 | 0.930141 | 0.966899 | 3.329 | 1055.658 |
| MetaQA | Beam-100 | 0.897782 | 0.942106 | 0.999130 | 57.978 | 0.000 |
| MQuAKE-ST Single | Greedy | 0.886303 | 0.886303 | 0.886303 | 1.000 | 0.000 |
| MQuAKE-ST Single | Top-2 | 0.886303 | 0.888298 | 0.890293 | 1.027 | 399.801 |
| MQuAKE-ST Single | Top-4 | 0.886303 | 0.888298 | 0.890293 | 1.030 | 799.269 |
| MQuAKE-ST Single | NumPy unrestricted | 0.886303 | 0.888298 | 0.890293 | 1.030 | 2115.297 |
| MQuAKE-ST Single | Beam-100 | 0.886303 | 0.922135 | 0.994016 | 52.735 | 0.000 |
| MQuAKE-ST Multi | Greedy | 0.856322 | 0.856322 | 0.856322 | 1.000 | 0.000 |
| MQuAKE-ST Multi | Top-2 | 0.856322 | 0.860536 | 0.865517 | 1.055 | 399.744 |
| MQuAKE-ST Multi | Top-4 | 0.856322 | 0.860536 | 0.865517 | 1.059 | 797.637 |
| MQuAKE-ST Multi | NumPy unrestricted | 0.856322 | 0.860536 | 0.865517 | 1.059 | 2201.977 |
| MQuAKE-ST Multi | Beam-100 | 0.856322 | 0.889130 | 0.977011 | 56.526 | 0.000 |

### 13.4 Complete primary readouts

The next two tables add the required rollout, correct-candidate, candidate-budget, and fixed-rank fields. `Mean correct slots` counts correct terminal slots among 100; `mean unique correct` counts distinct correct terminal entity IDs per question.

| Dataset | Mode | Rollout success | MRR given coverage | Mean correct slots | Mean unique correct terminals |
|---|---|---:|---:|---:|---:|
| Kinship | Greedy | 0.960396 | 1.000000 | 96.040 | 0.960 |
| Kinship | Top-2 | 0.966337 | 0.994949 | 96.634 | 0.980 |
| Kinship | Top-4 | 0.965248 | 0.990000 | 96.525 | 0.990 |
| Kinship | NumPy unrestricted | 0.962871 | 0.985000 | 96.287 | 0.990 |
| Kinship | Beam-100 | 0.197327 | 0.977723 | 19.733 | 1.000 |
| MetaQA | Greedy | 0.897910 | 1.000000 | 89.791 | 0.898 |
| MetaQA | Top-2 | 0.895679 | 0.965779 | 89.568 | 1.588 |
| MetaQA | Top-4 | 0.895019 | 0.962470 | 89.502 | 2.194 |
| MetaQA | NumPy unrestricted | 0.894701 | 0.961983 | 89.470 | 2.893 |
| MetaQA | Beam-100 | 0.267877 | 0.942926 | 26.788 | 7.332 |
| MQuAKE-ST Single | Greedy | 0.886303 | 1.000000 | 88.630 | 0.886 |
| MQuAKE-ST Single | Top-2 | 0.885246 | 0.997760 | 88.525 | 0.890 |
| MQuAKE-ST Single | Top-4 | 0.885246 | 0.997760 | 88.525 | 0.890 |
| MQuAKE-ST Single | NumPy unrestricted | 0.885246 | 0.997760 | 88.525 | 0.890 |
| MQuAKE-ST Single | Beam-100 | 0.139621 | 0.927686 | 13.962 | 0.994 |
| MQuAKE-ST Multi | Greedy | 0.856322 | 1.000000 | 85.632 | 0.856 |
| MQuAKE-ST Multi | Top-2 | 0.854793 | 0.994245 | 85.479 | 0.872 |
| MQuAKE-ST Multi | Top-4 | 0.854805 | 0.994245 | 85.480 | 0.874 |
| MQuAKE-ST Multi | NumPy unrestricted | 0.854805 | 0.994245 | 85.480 | 0.874 |
| MQuAKE-ST Multi | Beam-100 | 0.163391 | 0.910050 | 16.339 | 2.476 |

| Dataset | Mode | Candidate slots | Beam width | Mean / median unique | Unique fraction | Fixed-rank bits/q | Stochastic payload/q | Truncation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Kinship | Greedy | 100 | — | 1.000 / 1 | 0.010000 | 0.000 | 0.000 | 0.000% |
| Kinship | Top-2 | 100 | — | 1.158 / 1 | 0.011584 | 300.000 | 300.000 | 0.000% |
| Kinship | Top-4 | 100 | — | 1.257 / 1 | 0.012574 | 600.000 | 600.000 | 0.000% |
| Kinship | NumPy unrestricted | 100 | — | 1.327 / 1 | 0.013267 | 800.287 | 800.287 | 0.000% |
| Kinship | Beam-100 | 100 | 100 | 9.109 / 10 | 0.091089 | — | 0.000 | 0.000% |
| MetaQA | Greedy | 100 | — | 1.000 / 1 | 0.010000 | 0.000 | 0.000 | 1.917% |
| MetaQA | Top-2 | 100 | — | 1.849 / 2 | 0.018495 | 300.000 | 300.000 | 1.917% |
| MetaQA | Top-4 | 100 | — | 2.562 / 2 | 0.025622 | 568.814 | 568.814 | 1.917% |
| MetaQA | NumPy unrestricted | 100 | — | 3.329 / 2 | 0.033292 | 1055.658 | 1055.658 | 1.917% |
| MetaQA | Beam-100 | 100 | 100 | 57.978 / 63 | 0.579776 | — | 0.000 | 7.905% |
| MQuAKE-ST Single | Greedy | 100 | — | 1.000 / 1 | 0.010000 | 0.000 | 0.000 | 5.170% |
| MQuAKE-ST Single | Top-2 | 100 | — | 1.027 / 1 | 0.010266 | 399.801 | 399.801 | 5.147% |
| MQuAKE-ST Single | Top-4 | 100 | — | 1.030 / 1 | 0.010299 | 799.269 | 799.269 | 5.147% |
| MQuAKE-ST Single | NumPy unrestricted | 100 | — | 1.030 / 1 | 0.010299 | 2115.297 | 2115.297 | 5.147% |
| MQuAKE-ST Single | Beam-100 | 100 | 100 | 52.735 / 52 | 0.527347 | — | 0.000 | 9.352% |
| MQuAKE-ST Multi | Greedy | 100 | — | 1.000 / 1 | 0.010000 | 0.000 | 0.000 | 18.391% |
| MQuAKE-ST Multi | Top-2 | 100 | — | 1.055 / 1 | 0.010552 | 399.744 | 399.744 | 18.377% |
| MQuAKE-ST Multi | Top-4 | 100 | — | 1.059 / 1 | 0.010586 | 797.637 | 797.637 | 18.377% |
| MQuAKE-ST Multi | NumPy unrestricted | 100 | — | 1.059 / 1 | 0.010586 | 2201.977 | 2201.977 | 18.377% |
| MQuAKE-ST Multi | Beam-100 | 100 | 100 | 56.526 / 56 | 0.565264 | — | 0.000 | 16.570% |

### 13.5 Candidate-diversity and answer-coverage diagnostics

Beam width 100 means 100 retained beam slots, not 100 unique valid paths. Low-degree states can retain repeated/filler low-score branches under the pinned implementation. Actual diversity is therefore described only by the measured terminal quantities below.

| Dataset | Mode | Candidate coverage | Mean unique | Median unique | Unique fraction | MRR given coverage |
|---|---|---:|---:|---:|---:|---:|
| Kinship | Greedy | 0.960396 | 1.000 | 1 | 0.010000 | 1.000000 |
| Kinship | Top-2 | 0.980198 | 1.158 | 1 | 0.011584 | 0.994949 |
| Kinship | Top-4 | 0.990099 | 1.257 | 1 | 0.012574 | 0.990000 |
| Kinship | NumPy unrestricted | 0.990099 | 1.327 | 1 | 0.013267 | 0.985000 |
| Kinship | Beam-100 | 1.000000 | 9.109 | 10 | 0.091089 | 0.977723 |
| MetaQA | Greedy | 0.897910 | 1.000 | 1 | 0.010000 | 1.000000 |
| MetaQA | Top-2 | 0.960965 | 1.849 | 2 | 0.018495 | 0.965779 |
| MetaQA | Top-4 | 0.966234 | 2.562 | 2 | 0.025622 | 0.962470 |
| MetaQA | NumPy unrestricted | 0.966899 | 3.329 | 2 | 0.033292 | 0.961983 |
| MetaQA | Beam-100 | 0.999130 | 57.978 | 63 | 0.579776 | 0.942926 |
| MQuAKE-ST Single | Greedy | 0.886303 | 1.000 | 1 | 0.010000 | 1.000000 |
| MQuAKE-ST Single | Top-2 | 0.890293 | 1.027 | 1 | 0.010266 | 0.997760 |
| MQuAKE-ST Single | Top-4 | 0.890293 | 1.030 | 1 | 0.010299 | 0.997760 |
| MQuAKE-ST Single | NumPy unrestricted | 0.890293 | 1.030 | 1 | 0.010299 | 0.997760 |
| MQuAKE-ST Single | Beam-100 | 0.994016 | 52.735 | 52 | 0.527347 | 0.927686 |
| MQuAKE-ST Multi | Greedy | 0.856322 | 1.000 | 1 | 0.010000 | 1.000000 |
| MQuAKE-ST Multi | Top-2 | 0.865517 | 1.055 | 1 | 0.010552 | 0.994245 |
| MQuAKE-ST Multi | Top-4 | 0.865517 | 1.059 | 1 | 0.010586 | 0.994245 |
| MQuAKE-ST Multi | NumPy unrestricted | 0.865517 | 1.059 | 1 | 0.010586 | 0.994245 |
| MQuAKE-ST Multi | Beam-100 | 0.977011 | 56.526 | 56 | 0.565264 | 0.910050 |

Top-2/Top-4/NumPy produce little terminal diversity on Kinship (1.158/1.257/1.327 unique terminals on average) and almost none on MQuAKE (at most 1.030 Single and 1.059 Multi). MetaQA is more responsive, rising from 1.849 to 3.329 across the stochastic supports. Beam produces substantially more actual terminal diversity: 9.109 Kinship, 57.978 MetaQA, 52.735 Single, and 56.526 Multi, with corresponding coverage 1.000000, 0.999130, 0.994016, and 0.977011.

Across these decoder families, greater measured diversity accompanies greater coverage. Coverage explains the direction of most beam MRR gains, but diversity alone is not sufficient to determine ranking quality: beam's MRR conditional on coverage is lower than the stochastic alternatives on all four datasets. This is a diagnostic association, not a causal identification.

### 13.6 Deterministic beam versus stochastic execution

#### Kinship

The full 101-question result is mixed (Case C). Beam-100 has Hits@1 `0.960396`, MRR `0.977723`, complete candidate coverage, conditional MRR `0.977723`, and mean/median terminal diversity `9.109/10`. It exceeds Top-2 and NumPy MRR by `0.002475`, but falls below Top-4 by `0.002475`; Top-4 also has one additional Hits@1 question. Beam's higher coverage than Top-4 (`+0.009901`) is offset by lower conditional ranking (`−0.012277`).

The earlier 64-question smoke suggested beam might be the best ranked decoder (`MRR=0.980469`). The full result confirms that beam is competitive and much more diverse, but overturns the stronger smoke impression that it leads all modes: full-data Top-4 is slightly better. Historical seed-42 stochastic values are reproduced exactly, and the earlier five-seed variation still counsels against overinterpreting one-question Kinship differences.

#### MetaQA

Beam-100 has Hits@1 `0.897782`, MRR `0.942106`, candidate coverage `0.999130`, conditional MRR `0.942926`, mean/median unique terminals `57.978/63`, and unique fraction `0.579776`. It exceeds NumPy MRR by `0.011965`, Top-4 by `0.012134`, and Top-2 by `0.014026`, while matching Top-2/NumPy Hits@1 and trailing greedy by five of 39,093 questions.

Relative to NumPy, beam gains `0.032231` coverage but loses `0.019057` conditional MRR, so its net ranking advantage is primarily candidate exposure rather than better ordering once an answer is present. The derived quantity

```text
rho_beam = (MRR_beam - MRR_greedy) / (MRR_numpy - MRR_greedy) = 1.371233
```

means beam obtains 137.1% of the seed-42 NumPy-over-greedy MRR gain. This ratio is derived here, not emitted by the evaluator. Thus deterministic Beam-100 more than recovers the previously observed MetaQA stochastic MRR gain with zero incremental stochastic-realization payload under the shared-side-information accounting.

#### MQuAKE-ST Single

Beam-100 preserves the common Hits@1 `0.886303` but reaches MRR `0.922135`, versus `0.888298` for every stochastic mode (`+0.033837`). Coverage rises from `0.890293` to `0.994016`, while conditional MRR falls from `0.997760` to `0.927686`; mean unique terminals rise from about `1.03` to `52.735`. This is a strong coverage-driven beam advantage (Case A with a coverage/ranking tradeoff), not evidence for stochastic realization.

#### MQuAKE-ST Multi

Beam-100 preserves the common Hits@1 `0.856322` and reaches MRR `0.889130`, versus `0.860536` for every stochastic mode (`+0.028593`). Coverage rises from `0.865517` to `0.977011`, conditional MRR falls from `0.994245` to `0.910050`, and mean unique terminals rise from about `1.06` to `56.526`. This is again a coverage-driven deterministic-beam advantage (Case A with a coverage/ranking tradeoff).

### 13.7 Uniform NumPy unrestricted reference

The new primary campaign supplies a uniform NumPy unrestricted `R=100` reference. The preserved historical TensorFlow rows remain valid regression references.

| Dataset | Metric | Historical TF | New NumPy | NumPy − TF |
|---|---|---:|---:|---:|
| MQuAKE-ST Single | Hits@1 | 0.886303 | 0.886303 | 0.000000 |
| MQuAKE-ST Single | MRR | 0.888852 | 0.888298 | −0.000554 |
| MQuAKE-ST Single | Rollout success | 0.885339 | 0.885246 | −0.000093 |
| MQuAKE-ST Single | Full-support fixed-rank bits/q | 2115.299 | 2115.297 | −0.002 |
| MQuAKE-ST Multi | Hits@1 | 0.856322 | 0.856322 | 0.000000 |
| MQuAKE-ST Multi | MRR | 0.860153 | 0.860536 | +0.000383 |
| MQuAKE-ST Multi | Rollout success | 0.854897 | 0.854805 | −0.000092 |
| MQuAKE-ST Multi | Full-support fixed-rank bits/q | 2202.123 | 2201.977 | −0.146 |

The backend change leaves Hits@1 identical and changes MRR/rollout success by less than `0.0006`; the full-support fixed-rank reference is also effectively unchanged. It does not alter the previous MQuAKE conclusion of a near-deterministic stochastic policy and small stochastic ranking gains. Backend-specific sampled surprisal/Shannon diagnostics need not match because the action draws differ.

### 13.8 MQuAKE cap-200 versus cap-512 sensitivity

Values are cap 200 → cap 512. Truncation is measured over the states/slots visited by each decoder, so beam and stochastic truncation fractions can differ at cap 200.

| Dataset | Mode | Hits@1 | MRR | Coverage | MRR given coverage | Mean unique | Rollout success | Truncation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MQuAKE-ST Single | Greedy | 0.886303→0.888963 | 0.886303→0.888963 | 0.886303→0.888963 | 1.000000→1.000000 | 1.000→1.000 | 0.886303→0.888963 | 5.170%→0.000% |
| MQuAKE-ST Single | Top-2 | 0.886303→0.888963 | 0.888298→0.891290 | 0.890293→0.893617 | 0.997760→0.997396 | 1.027→1.027 | 0.885246→0.887819 | 5.147%→0.000% |
| MQuAKE-ST Single | Top-4 | 0.886303→0.888963 | 0.888298→0.891290 | 0.890293→0.893617 | 0.997760→0.997396 | 1.030→1.031 | 0.885246→0.887819 | 5.147%→0.000% |
| MQuAKE-ST Single | NumPy unrestricted | 0.886303→0.888963 | 0.888298→0.891290 | 0.890293→0.893617 | 0.997760→0.997396 | 1.030→1.031 | 0.885246→0.887819 | 5.147%→0.000% |
| MQuAKE-ST Single | Beam-100 | 0.886303→0.888963 | 0.922135→0.926025 | 0.994016→0.996676 | 0.927686→0.929114 | 52.735→51.549 | 0.139621→0.151489 | 9.352%→0.000% |
| MQuAKE-ST Multi | Greedy | 0.856322→0.840230 | 0.856322→0.840230 | 0.856322→0.840230 | 1.000000→1.000000 | 1.000→1.000 | 0.856322→0.840230 | 18.391%→0.000% |
| MQuAKE-ST Multi | Top-2 | 0.856322→0.840230 | 0.860536→0.848467 | 0.865517→0.857471 | 0.994245→0.989500 | 1.055→1.062 | 0.854793→0.839966 | 18.377%→0.000% |
| MQuAKE-ST Multi | Top-4 | 0.856322→0.840230 | 0.860536→0.848467 | 0.865517→0.857471 | 0.994245→0.989500 | 1.059→1.068 | 0.854805→0.839943 | 18.377%→0.000% |
| MQuAKE-ST Multi | NumPy unrestricted | 0.856322→0.840230 | 0.860536→0.848467 | 0.865517→0.857471 | 0.994245→0.989500 | 1.059→1.068 | 0.854805→0.839943 | 18.377%→0.000% |
| MQuAKE-ST Multi | Beam-100 | 0.856322→0.840230 | 0.889130→0.879428 | 0.977011→0.979310 | 0.910050→0.898007 | 56.526→50.225 | 0.163391→0.227828 | 16.570%→0.000% |

| Dataset | Mode | Fixed/stochastic payload 200→512 (bits/q) | Effective beam width |
|---|---|---:|---:|
| MQuAKE-ST Single | Greedy | 0.000→0.000 | — |
| MQuAKE-ST Single | Top-2 | 399.801→400.000 | — |
| MQuAKE-ST Single | Top-4 | 799.269→799.668 | — |
| MQuAKE-ST Single | NumPy unrestricted | 2115.297→2125.943 | — |
| MQuAKE-ST Single | Beam-100 | NA fixed-rank; 0 stochastic → NA fixed-rank; 0 stochastic | 100→100 |
| MQuAKE-ST Multi | Greedy | 0.000→0.000 | — |
| MQuAKE-ST Multi | Top-2 | 399.744→399.744 | — |
| MQuAKE-ST Multi | Top-4 | 797.637→797.637 | — |
| MQuAKE-ST Multi | NumPy unrestricted | 2201.977→2260.137 | — |
| MQuAKE-ST Multi | Beam-100 | NA fixed-rank; 0 stochastic → NA fixed-rank; 0 stochastic | 100→100 |

1. **Beam performance:** removing truncation changes Single modestly: beam Hits@1 gains four questions (`+0.002660`), MRR rises `+0.003890`, and coverage rises `+0.002660`. Multi remains materially cap-sensitive: beam Hits@1 loses 14 questions (`−0.016092`), MRR falls `−0.009702`, mean diversity falls by `6.301`, and conditional MRR falls `−0.012043`, despite coverage increasing `+0.002299`.
2. **Stochastic performance:** Single Top-2/Top-4/NumPy MRR rises `+0.002992`, while Multi stochastic MRR falls `−0.012069` and Hits@1 falls the same 14 questions as greedy/beam. These are common cap-induced baseline changes, not newly revealed stochastic gains.
3. **Ordering:** cap 512 does not change the decoder ordering relevant to the reviewer question. Beam remains substantially above all stochastic modes in MRR on both splits.
4. **Stochastic advantage:** none appears. The beam-minus-stochastic MRR gap is about `+0.033837` at cap 200 and `+0.034735` at cap 512 for Single; it is `+0.028593` and `+0.030960` for Multi.
5. **Relative cap sensitivity:** Multi remains more cap-sensitive in absolute task performance. Its unusual response confirms that cap 512 is only an evaluation sensitivity of a checkpoint trained under cap 200; it is not automatically a better inference condition.

### 13.9 Communication interpretation

Greedy and deterministic beam both have zero incremental stochastic action-realization payload under synchronized policy, state, local action interface/order, and tie-breaking. Top-2/Top-4 values are fixed-rank sampled-action payloads summed over all 100 trajectories and hops. NumPy is the full-support stochastic local-rank reference. Beam fixed-rank/source-code fields remain null because deterministic branch retention is not a sequence of sampled action messages.

Beam nevertheless incurs additional computation, beam-state maintenance, candidate processing, and stringent synchronization assumptions. Those costs were not measured and lie outside this conditional action-message accounting. The present experiment therefore compares ranking utility against stochastic action payload, not total system cost or equal compute.

### 13.10 Scientific conclusion and reviewer hypothesis

The strict four-dataset result is **mixed (Case C), but it strongly supports the reviewer hypothesis**. Deterministic beam exceeds every stochastic mode in MRR on MetaQA and both MQuAKE splits; on Kinship it exceeds Top-2 and NumPy but trails Top-4 by only `0.002475`. Thus stochastic action realization is not uniquely responsible for the sampled-candidate ranking gains. Much—and on three datasets more than all—of the stochastic MRR improvement is obtainable through deterministic diverse decoding with zero incremental stochastic-realization messages under the tested shared-side-information assumptions.

The decomposition is consistent across datasets: beam exposes many more terminal entities and substantially raises answer coverage, while ranking answers worse conditional on coverage. Beam wins overall when its coverage gain outweighs that conditional-ranking loss. Kinship Top-4 is the one case where the conditional-ranking tradeoff slightly favors stochastic decoding. These diagnostics support a candidate-exposure explanation within the tested decoders, but do not establish causality or optimality of beam search.

### 13.11 High-level implications for future paper revision

- The prior claim that stochastic branching itself supplies the `R=100` ranking utility is stale and must be qualified. The evidence now supports **diverse decoding**, not stochastic realization specifically, as the central mechanism.
- A unique communication benefit from stochastic branching cannot be claimed from these results. Within the stochastic family, Top-2/Top-4 still reduce action payload relative to NumPy for similar ranking utility, but deterministic beam often delivers greater MRR with zero stochastic-realization payload at a different and unmeasured computational cost.
- The scientifically supported framing is a communication/computation/candidate-diversity tradeoff: greedy is cheap and top-1 competitive; stochastic Top-K offers controlled sampled branching; deterministic beam offers much broader candidate exposure without stochastic action messages but with higher search/state-processing cost.
- The historical observation that candidate diversity was only a plausible, unmeasured explanation is now superseded: terminal diversity, answer coverage, and conditional ranking are measured directly. The historical final verdict that no additional core experiment was needed was superseded by Section 12 and is now empirically resolved by this section.

### 13.12 Remaining limitations

- The primary stochastic comparison is seed 42. Kinship and MetaQA seed-42 stochastic values exactly reproduce their existing five-seed campaign rows, but the new beam comparison itself is not a five-seed study. Beam is deterministic under fixed execution state/tie behavior.
- `R=100` MRR remains MINERVA sampled-candidate MAX-pool MRR, not full-KG entity-ranking MRR.
- Beam width 100 is 100 retained slots, not 100 unique valid paths; actual unique fractions range from `0.091089` to `0.579776`.
- Decoder compute, memory, latency, candidate-processing, and synchronization-establishment costs are not measured or matched.
- Cap-200 truncation is decoder-state dependent; MetaQA remains truncated, and MQuAKE cap 512 is evaluation-only.
- Zero stochastic payload is conditional on identical policy/state/local-interface/order/tie-breaking and is not zero total communication.
- No new multi-seed experiment is required to establish beam determinism or the primary coverage/ranking decomposition, but the single-seed stochastic rows should not be described as new multi-seed evidence.
