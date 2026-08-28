# Codex Implementation Brief — MINERVA-InfoCost No-Retraining Revision

**Target repository:** `https://github.com/HernandezEduin/MINERVA-InfoCost`  
**Default branch when inspected:** `master`  
**MINERVA submodule commit when inspected:** `9bf1ae998d14471c3f7c31f70969d0bbf9873329`  
**Prepared:** 2026-08-27  
**Updated:** 2026-08-28 after completion of the no-retraining follow-up experiments
**Scope:** Evaluation-only changes. **Do not retrain MINERVA. Do not modify pretrained checkpoints.**

---

## 1. Objective

The rejected SPAWC version showed that the entropy of a learned, non-uniform MINERVA action policy can be much smaller than a local identifier cost such as `log2(|A_t|)`. The SPAWC reviewers identified five core weaknesses:

1. The entropy/source-coding connection follows directly from Shannon coding.
2. The paper did not show what happens to task performance under a real communication budget.
3. For a non-uniform policy, `H(pi) <= log2(|A_t|)` is expected, so the uniform/local-index baseline is weak.
4. The communication evaluation was idealized and did not cleanly separate entropy from realizable integer code length.
5. The policy was learned offline and only analyzed after training, so no rate–accuracy trade-off was established.

The current ICASSP draft has updated results, including MQuAKE-ST single-answer/multi-answer settings and PED, but its central entropy-vs-identifier experiment is still too close to the rejected SPAWC study.

### No-retraining goal

Using the **existing pretrained checkpoints**, add evaluation functionality that answers:

> **How does restricting the amount of action information communicated at each hop affect reasoning accuracy and path fidelity?**

The primary deliverable is a **rate/budget vs. task-performance evaluation** for the current pretrained policies.

> **Final no-retraining status:** P0 and the follow-up implementation and experiments are complete. The evaluator supports explicit `R=1` execution, total per-question communication for `R>1` ensembles, backend-matched `numpy_policy` sampling, and evaluation-only action-cap overrides. MQuAKE cap sensitivity and five-seed Kinship and MetaQA `R=100` studies are complete. No further core experiment is currently required. Detailed artifact paths and quantitative results are tracked in `codex_ref/infocost_experiment_results_20260828.md`.

---

## 2. Hard constraints

- **No retraining.**
- **Do not change checkpoint weights.**
- **Do not add communication-aware loss terms.**
- **Do not change or regenerate datasets.**
- **Use the current ICASSP-era configs, not old SPAWC settings.**
- **Keep `use_beam=False` for the main experiments.**
- **Do not modify the `minerva/` submodule unless absolutely unavoidable.**
- Preserve current behavior when new features are disabled.
- The unrestricted InfoCost evaluation must remain runnable.
- Do not silently change the definition of MINERVA Hits@1 or PED.
- New experiments must be reproducible from a seed.
- Never estimate the task-agnostic statistical baseline from test/evaluation action choices.

---

## 3. Repository state and relevant files

```text
MINERVA-InfoCost/
├── code/
│   ├── evaluation_infocost.py
│   └── policy_entropy/
│       ├── eval.py
│       ├── metrics.py
│       ├── artifacts.py
│       └── plotting.py
├── configs/
│   ├── kinshiphinton.yaml
│   ├── metaqa.yaml
│   ├── mquake_st_single.yaml
│   └── mquake_st_multi.yaml
├── minerva/
└── run_infocost.sh
```

### Existing flow

`code/evaluation_infocost.py`:

1. calls upstream `read_options()`;
2. loads vocabularies;
3. creates `EmbeddingServer`;
4. creates `TrainerNLQ`;
5. restores `options["model_load_dir"]`;
6. runs `analyze_policy_entropy_testset(...)`;
7. saves entropy artifacts/plots.

`run_infocost.sh` sets `PYTHONPATH`, selects CPU/GPU, and launches that script.

### Important CLI constraint

Upstream `minerva/code/options.py::read_options()` uses `argparse.parse_args()` and rejects unknown arguments. Its YAML loader also rejects unknown YAML keys.

Therefore, do **not** add rate-only keys to the existing YAMLs unless upstream options are changed.

Preferred approach for a new `evaluation_rate_sweep.py`:

1. use a custom `argparse` pre-parser with `parse_known_args()`;
2. consume/remove rate-specific CLI options;
3. replace `sys.argv` with only the remaining MINERVA arguments;
4. then call upstream `read_options()`.

This avoids submodule changes.

---

## 4. Current configs are the source of truth

### KINSHIP

```text
test_rollouts: 100
max_num_actions: 100
use_beam: False
use_full_graph: True
use_directed_graph: True
path_length: 3
question_format: full_text
```

### METAQA

```text
test_rollouts: 100
max_num_actions: 200
use_beam: False
use_full_graph: True
use_directed_graph: False
path_length: 3
question_format: full_text
```

### MQUAKE-ST single answer

```text
test_rollouts: 100
max_num_actions: 200
use_beam: False
use_full_graph: True
use_directed_graph: True
path_length: 4
raw_QAData_path: .../mquake_sa_qa_nhop.csv
```

### MQUAKE-ST multi answer

```text
test_rollouts: 100
max_num_actions: 200
use_beam: False
use_full_graph: True
use_directed_graph: True
path_length: 4
raw_QAData_path: .../mquake_ma_qa_nhop.csv
```

Do not revert these to the rejected SPAWC configuration.

---

## 5. Current policy-entropy implementation

### `code/policy_entropy/eval.py`

`collect_policy_entropy_single_episode(...)` currently:

1. obtains `next_relations`, `next_entities`, and `current_entities`;
2. feeds them into the pretrained test graph;
3. fetches `trainer.test_logits`, `trainer.test_action_idx`, `trainer.chosen_relation`, and updated recurrent memory;
4. treats `test_logits` as log-softmax probabilities;
5. computes entropy, sampled-action surprisal, valid action count, ideal identifier bits, fixed identifier bits, and savings;
6. advances with `state = episode(test_action_idx)`.

The evaluator already returns final entities.

### `code/policy_entropy/metrics.py`

Already provides:

```python
entropy_bits_from_log_probs(...)
action_surprisal_bits_from_log_probs(...)
count_valid_action(...)
ideal_uniform_identifier_bits(...)
fixed_width_identifier_bits(...)
```

In particular, sampled-action surprisal is already:

```text
-log2 pi(a_t | s_t)
```

### Safe action override

The current step's `trainer.test_state` is produced from the **previous relation/current entity** and the current recurrent memory. The newly selected action is then used to update the environment and becomes `previous_relation` on the next hop.

Therefore greedy/Top-K evaluation can:

1. fetch `test_logits` and `test_state`;
2. select a different action in NumPy;
3. recompute `chosen_relation` from the current local action array;
4. set `previous_relation` to that overridden relation;
5. call `episode(overridden_action_idx)`.

Do **not** continue using TensorFlow's original `chosen_relation` after overriding the action.

---

## 6. Preserve MINERVA Hits@1 and PED semantics

MINERVA Hits@1 is **not rollout success rate**.

With `pool="max"`, upstream `TrainerNLQ.test()`:

1. accumulates cumulative path log probability;
2. reshapes to `[num_questions, test_rollouts]`;
3. sorts rollouts by descending cumulative log probability;
4. searches the ranked rollouts while counting unique terminal entities;
5. determines the rank of the first correct answer;
6. Hits@1 is true only when the first correct answer has rank 1.

For path metrics/PED, upstream uses the **highest-scoring rollout**:

```python
r = sorted_indx[b][0]
```

then reconstructs/cleans that path and computes PED.

### Requirement

The custom rate evaluator must reproduce this behavior for:

- Hits@1;
- optionally MRR;
- PED/path metrics.

It is fine to also report:

```text
rollout_success_rate = mean(answer_hits)
```

but it must not be called Hits@1.

---

## 7. Core experiment A — greedy / zero-action-message baseline

Add:

```text
action_mode = "greedy"
```

At each hop:

```python
action_idx = argmax(valid log probability)
```

### Interpretation

Under synchronized side information, if both sides have:

- the same policy;
- the same state/task;
- the same local action set and ordering;

then deterministic argmax can be reproduced with **0 action-payload bits**.

This is a required sanity check.

### Report

- Hits@1;
- PED where available;
- MRR if easy to preserve;
- rollout success rate;
- mean original-policy entropy at visited states;
- mean valid action count;
- action payload budget = `0 bits/hop` under the synchronized-side-information interpretation.

If greedy preserves essentially all unrestricted performance, flag it clearly. Do not hide it.

---

## 8. Core experiment B — hard per-hop Top-K budget

Add:

```text
action_mode = "topk"
```

For support size `K`:

1. start from pretrained log probabilities;
2. exclude invalid/padded actions;
3. retain only the `K` highest-probability valid actions;
4. if fewer than K valid actions exist, retain all valid actions;
5. renormalize;
6. sample from the truncated policy;
7. advance MINERVA with the sampled action.

Formally:

```text
A_t^(K) = TopK actions under pi(.|s_t)

pi_K(a|s_t) =
  pi(a|s_t) / sum_{a' in A_t^(K)} pi(a'|s_t), if a in A_t^(K)
  0, otherwise
```

Use:

```text
K = 2^B
```

Recommended sweep:

```text
K = 1, 2, 4, 8, 16, 32, 64, 128, unrestricted
B = 0, 1, 2, 3, 4, 5, 6, 7, unrestricted
```

K=1 should coincide with greedy.

### Keep hard budget separate from entropy coding

Top-K + fixed local rank gives a hard per-step support/index bound.

Entropy is an expected source-coding quantity.

Do not merge the two definitions.

### Required outputs per K

- K;
- nominal budget `B=log2(K)`;
- Hits@1;
- MRR if implemented;
- PED where available;
- rollout success rate;
- mean valid actions;
- mean effective retained support;
- mean fixed-width bits actually required:
  `ceil(log2(min(K, |A_t|)))`;
- mean entropy of the truncated execution policy;
- mean sampled surprisal under that policy;
- mean path-level cost;
- success-conditioned cost.

### Main plots

```text
Hits@1 vs nominal bit budget
PED vs nominal bit budget
Hits@1 vs empirical mean fixed-rank bits
```

---

## 9. Top-K sampling details

Do not use `trainer.test_action_idx` for Top-K.

Instead:

1. fetch `trainer.test_logits`;
2. construct truncated log probabilities in NumPy;
3. sample with a seeded NumPy RNG;
4. override action index;
5. recompute chosen relation.

Suggested helper:

```python
def select_actions_from_log_probs(
    log_probs,
    valid_mask,
    mode,
    top_k=None,
    rng=None,
):
    ...
```

Requirements:

- stable log-space renormalization;
- invalid actions remain impossible;
- row-wise `K = min(K, valid_count)`;
- selected action must always be valid;
- selected Top-K action must belong to retained support;
- greedy tie-breaking must be deterministic/documented.

Use:

```python
np.random.default_rng(rate_seed)
```

and save the seed in metadata.

---

## 10. Rank rollouts using the execution policy

Maintain:

```text
path_log_prob = sum_t log pi_execution(a_t | s_t)
```

where `pi_execution` is:

- original policy in original mode;
- truncated/renormalized policy for Top-K;
- deterministic policy for greedy if a score is needed.

For Top-K, do **not** rank rollouts using the untruncated original-policy score.

The cumulative execution-policy score must drive MINERVA-compatible `pool="max"` ranking.

---

## 11. Core experiment C — task-agnostic non-uniform coding baseline

Keep the uniform identifier baselines only as reference.

Add a stronger, **task-agnostic** non-uniform probability model.

### Recommended structural prior

Estimate global relation frequencies from the graph/action representation available to the evaluator, preferably via:

```python
trainer.environment.grapher.array_store
```

while excluding invalid relation IDs.

Use smoothing:

```text
p_global(r) proportional to count(r) + alpha
```

with documented `alpha`, e.g. `1.0`.

This avoids fitting from test rollout choices.

### Local distribution

For unique valid local relations `R(s)`:

```text
q_R(r|s) =
  p_global(r) / sum_{r' in R(s)} p_global(r')
```

If relation r occurs in `m_r` local actions:

```text
q(a=(r,e')|s) = q_R(r|s) / m_r
```

This uses graph statistics and the local action set but **not the question embedding**.

### Compute

For the action selected by MINERVA:

```text
task-conditioned surprisal:
  -log2 pi(a_t | s_t, z)

task-agnostic surprisal:
  -log2 q(a_t | s_t)
```

Also compute state-wise:

```text
H(pi,q) = -sum_a pi(a|s,z) log2 q(a|s)
D_KL(pi || q) = H(pi,q) - H(pi)
```

This directly measures the benefit of task conditioning beyond generic non-uniform graph statistics.

### No leakage

Do not fit q from evaluation/test selected actions.

If counts are built from `array_store` while `use_full_graph=True`, label q a **graph-structural prior**, not a train-only prior.

---

## 12. Core experiment D — operational integer source-code length

Add a realizable prefix-code upper-bound length:

```text
l_Shannon(a|s) = ceil(-log2 p(a|s))
```

for both:

- task-conditioned `pi`;
- task-agnostic `q`.

Keep separate fields:

```text
entropy_bits
sampled_surprisal_bits
shannon_integer_code_bits
```

Do not call entropy "actual transmitted bits."

For finite nonzero selected probabilities:

```text
surprisal <= shannon_integer_length < surprisal + 1
```

Allow 0 bits for a deterministic one-symbol support.

A full Huffman/arithmetic coder is optional P1; do not delay the main rate–accuracy experiment for it.

---

## 13. Core experiment E — cost conditioned on correctness

After the final hop:

```python
_, answer_hits = episode.get_reward()
```

Store hit mask as `[B, R]`.

Report:

```text
E[path entropy | success]
E[path entropy | failure]

E[path surprisal | success]
E[path surprisal | failure]

E[path integer code bits | success]
E[path integer code bits | failure]
```

Include subset counts and return null/NaN when a subset is empty.

---

## 14. Core experiment F — fixed horizon vs. gold hop count

Current execution uses fixed maximum horizon:

```text
KINSHIP: N=3
METAQA: N=3
MQUAKE-ST: N=4
```

while each question has gold hop count n.

Upstream exposes it through:

```python
episode.get_path_length(b)
```

Do **not** change execution horizon in this task.

Report two cost aggregations over the same executed trajectory:

```text
fixed-horizon:
  sum_{t=0}^{N-1} cost_t

gold-hop-masked diagnostic:
  sum_{t=0}^{n-1} cost_t
```

Use explicit names such as:

```text
mean_path_entropy_fixed_horizon_bits
mean_path_entropy_gold_hops_bits
```

The second is diagnostic, not the actual current execution cost.

---

## 15. Core experiment G — max_num_actions truncation diagnostics

Upstream graph construction truncates outgoing actions at the configured cap.

Current caps:

```text
KINSHIP: 100
METAQA: 200
MQUAKE-ST: 200
```

Do not modify the submodule just for this diagnostic.

Implement a top-level helper that attempts to recover raw candidate counts before the cap from the graph/triple source or another reliable non-destructive source.

Respect:

- directed vs undirected mode;
- inverse-relation filtering;
- NO_OP/special slots;
- actual graph used by the evaluator.

Report at minimum:

```text
max raw valid action count
fraction of visited states truncated
mean raw count for truncated visited states
mean retained count for truncated visited states
per-hop truncation rate
```

If raw degree cannot be recovered unambiguously, report the limitation instead of guessing.

Do not automatically increase `max_num_actions` in this task.

---

## 16. Recommended code organization

Prefer additive changes:

```text
code/
├── evaluation_infocost.py          # preserve behavior
├── evaluation_rate_sweep.py        # NEW
└── policy_entropy/
    ├── eval.py
    ├── metrics.py
    ├── rate_constraints.py         # NEW pure NumPy helpers
    ├── artifacts.py
    ├── plotting.py
    └── diagnostics.py              # optional
```

Add:

```text
run_rate_sweep.sh
```

mirroring `run_infocost.sh`.

`evaluation_rate_sweep.py` should:

1. pre-parse rate-only arguments;
2. strip them from `sys.argv`;
3. call upstream `read_options()`;
4. load `TrainerNLQ` once;
5. restore checkpoint once;
6. run all requested modes/K values;
7. save a combined sweep table.

---

## 17. Suggested CLI

Desired usage:

```bash
python code/evaluation_rate_sweep.py     --rate_top_k 1 2 4 8 16 32 64 128     --rate_include_unrestricted true     --rate_seed 42     --config_yaml configs/mquake_st_single.yaml
```

Custom pre-parser can consume:

```text
--rate_top_k
--rate_include_unrestricted
--rate_seed
--rate_compute_task_agnostic
--rate_compute_truncation_diagnostics
```

Then pass only remaining args to upstream `read_options()`.

Convenience wrapper:

```bash
bash run_rate_sweep.sh configs/mquake_st_single.yaml 0
```

with optional GPU ID like `run_infocost.sh`.

---

## 18. Performance metrics in the custom evaluator

Per rollout collect:

```text
path_log_prob
final_entity
answer_hit
per_step_entropy
per_step_surprisal
per_step_integer_code_bits
per_step_valid_action_count
per_step_effective_support
per_step_fixed_budget_bits
```

For path-fidelity datasets, keep temporary entity/relation histories long enough to compute top-ranked PED, but full trajectories do not have to be persisted.

### Per question

Using:

```text
sorted_rollouts = argsort(-path_log_prob)
```

reproduce upstream MAX pooling:

- search ranked rollouts;
- avoid double-penalizing duplicate terminal entities;
- determine first correct answer rank;
- Hits@1 if that rank is zero;
- MRR from the first correct rank.

For PED:

- choose highest-scoring rollout;
- reconstruct entity/relation path;
- use the same episode cleaning and edit-distance functions as upstream.

Also report:

```text
rollout_success_rate = mean(answer_hit)
```

under its own name.

---

## 19. Outputs

Recommended:

```text
<output_dir>/rate_sweep/
├── rate_sweep_summary.csv
├── rate_sweep_summary.json
├── rate_sweep_metadata.json
├── hits1_vs_budget.png
├── ped_vs_budget.png
├── hits1_vs_empirical_rate.png
└── task_conditioned_vs_agnostic.png
```

Minimum summary fields:

```text
dataset
mode
top_k
nominal_budget_bits
num_questions
num_rollouts
hits_at_1
mrr
ped
rollout_success_rate
mean_valid_actions
mean_effective_support
mean_fixed_budget_bits
mean_step_entropy_bits
mean_path_entropy_fixed_horizon_bits
mean_path_entropy_gold_hops_bits
mean_step_surprisal_bits
mean_path_surprisal_bits
mean_step_shannon_code_bits
mean_path_shannon_code_bits
mean_task_agnostic_surprisal_bits
mean_task_agnostic_shannon_bits
mean_policy_vs_task_agnostic_kl_bits
success_mean_path_code_bits
failure_mean_path_code_bits
truncated_state_fraction
seed
```

Use null/NaN for unavailable metrics.

---

## 20. Tests

Pure NumPy tests should not require GPU.

### P0

#### Valid action selection

- invalid/PAD actions are never selected;
- retained support is `min(K, valid_count)`;
- retained probabilities normalize to 1.

#### Greedy

- selects maximum valid probability;
- K=1 matches greedy;
- fixed-rank cost is 0 bits.

#### Top-K

- selected action is always retained;
- if `K >= valid_count`, distribution equals original valid distribution;
- fixed-rank bits never exceed nominal B.

#### Surprisal/code length

- selected surprisal is finite;
- Shannon length is integer-valued;
- `surprisal <= ceil(surprisal) < surprisal + 1`.

#### Task-agnostic prior

- local q sums to 1;
- all valid actions have nonzero probability after smoothing;
- invalid actions have zero probability;
- no question embedding or answer label is used.

#### Gold-hop mask

- masked nonnegative path cost cannot exceed fixed-horizon cost;
- when n=N they are equal.

### Regression mode

Add:

```text
action_mode = "tf_policy"
```

which uses the original `trainer.test_action_idx`.

It should reproduce the existing evaluator's entropy/statistics to numerical tolerance for the same checkpoint/data/seed.

Where feasible, compare its Hits@1/PED to `TrainerNLQ.test(beam=False)` in a controlled fresh run.

---

## 21. Scientific caveats to surface

### Greedy zero-rate issue

If synchronized endpoints can independently reproduce the deterministic action, greedy can require zero action-payload bits. Treat this as a scientific result, not a bug.

### Top-K is not entropy coding

Top-K defines a hard support/index budget; entropy is an expected source-coding quantity.

### Fractional entropy is not a packet length

Keep entropy, surprisal, and integer code length separate.

### Task-agnostic baseline

Its purpose is to account for generic non-uniform graph/action statistics without using the question.

### Full-graph prior

If built from `array_store` under `use_full_graph=True`, call it graph-structural.

### Action cap

The policy may already see a truncated graph; report this.

### Fixed horizon

Gold-hop-masked cost is diagnostic unless execution is changed.

---

## 22. Implementation sequence

### Phase 0 — reproduce current behavior

- verify root HEAD;
- verify submodule SHA;
- initialize submodule if needed;
- run existing InfoCost evaluation on the smallest available config;
- preserve the pre-change output.

### Phase 1 — pure NumPy rate helpers

Implement/test:

```python
valid_action_mask(...)
topk_renormalized_log_probs(...)
select_greedy(...)
sample_log_probs(...)
effective_fixed_rank_bits(...)
shannon_integer_code_length(...)
build_global_relation_prior(...)
task_agnostic_local_log_probs(...)
kl_bits(...)
```

### Phase 2 — parameterize episode collection

Suggested conceptual API:

```python
collect_policy_entropy_single_episode(
    ...,
    action_mode="tf_policy",
    top_k=None,
    rng=None,
    relation_prior=None,
    compute_performance=False,
    compute_gold_hop_cost=True,
)
```

Defaults must preserve existing behavior.

### Phase 3 — add performance metrics

- execution-policy cumulative path score;
- final entities;
- answer hits;
- MINERVA-compatible Hits@1;
- MRR if straightforward;
- top-ranked PED.

### Phase 4 — greedy and Top-K sweep

Run:

```text
K=1,2,4,8,16,32,64,128,unrestricted
```

### Phase 5 — task-agnostic coder

Add task-agnostic surprisal, Shannon length, cross entropy, KL.

### Phase 6 — correctness/horizon diagnostics

Add success/failure conditional costs and gold-hop-masked costs.

### Phase 7 — action-cap diagnostics

Quantify raw/capped degree if reliable.

### Phase 8 — outputs/plots

Generate combined CSV/JSON and rate-performance plots.

### Phase 9 — run all four current configs

```text
configs/kinshiphinton.yaml
configs/metaqa.yaml
configs/mquake_st_single.yaml
configs/mquake_st_multi.yaml
```

Do not automatically overwrite manuscript values.

---

## 23. P0 TODO checklist

### Repository safety

- [x] Verify current root HEAD.
- [x] Verify `minerva` submodule SHA.
- [x] Do not modify the submodule.
- [x] Preserve `evaluation_infocost.py` behavior.
- [x] Preserve current YAML configs.

### Rate utilities

- [x] Valid-action mask helper.
- [x] Greedy selection.
- [x] Top-K masking and stable renormalization.
- [x] Seeded sampling from log probabilities.
- [x] Effective fixed-rank bits.
- [x] Unit tests.

### Performance metrics

- [x] Cumulative log probability under the execution policy.
- [x] Final entities and answer hits.
- [x] MINERVA `pool="max"` Hits@1.
- [x] MRR.
- [x] Top-ranked PED.
- [x] Rollout success rate under a distinct name.

### Greedy diagnostic

- [x] Add greedy mode.
- [x] Verify K=1 matches greedy.
- [x] Report greedy Hits@1/PED.
- [x] Mark action-payload rate as 0 bits under synchronized side information.
- [x] Report the exact greedy-vs-unrestricted delta; no automatic "nearly unchanged" classification is applied because no threshold is specified.

### Top-K sweep

- [x] K = `1,2,4,8,16,32,64,128,unrestricted`.
- [x] Nominal `B=log2(K)`.
- [x] Empirical fixed-rank bits.
- [x] Hits@1-vs-budget plot.
- [x] PED-vs-budget plot.
- [x] Hits@1-vs-empirical-rate plot.

### Task-agnostic baseline

- [x] Smoothed global relation prior without test action labels.
- [x] Local task-agnostic q.
- [x] Task-agnostic selected-action surprisal.
- [x] Cross entropy/KL against task-conditioned policy.
- [x] Correctly label the prior as graph-structural.

### Operational code length

- [x] Integer Shannon length for task-conditioned policy.
- [x] Integer Shannon length for task-agnostic q.
- [x] Separate entropy, surprisal, and integer code length.

### Correctness/horizon

- [x] Rollout success mask.
- [x] Success-conditioned path cost.
- [x] Failure-conditioned path cost.
- [x] Fixed-horizon path cost.
- [x] Gold-hop-masked diagnostic cost.

### Action-cap diagnostics

- [x] Recover raw candidate degree and validate it against the capped action store.
- [x] Visited-state truncation rate.
- [x] Per-hop truncation rate.
- [x] Emit an explicit limitation if recovery or validation is unavailable.

### Outputs

- [x] Combined CSV summary.
- [x] JSON metadata/results.
- [x] Save seed/config/checkpoint identity.
- [x] Dedicated `rate_sweep/` output directory.
- [x] `run_rate_sweep.sh`.
- [x] Document usage.

---

## 24. P1 / optional TODO

Only after P0 works:

- [ ] Canonical Huffman lengths.
- [ ] Encode/decode round-trip on representative subset.
- [ ] Coding CPU latency.
- [ ] Bootstrap confidence intervals.
- [ ] Multiple evaluation seeds.
- [ ] Diagnostics:
  - `max_a pi(a|s)`
  - `2^H`
  - top1-top2 margin
- [ ] Per-hop rate–accuracy analysis.
- [ ] Plot uniform local index vs task-agnostic adaptive model vs task-conditioned model.

---

## 25. Out of scope

Do not implement unless separately requested:

- policy retraining;
- entropy-penalized fine-tuning;
- communication-aware RL;
- new datasets/models;
- dynamic policy synchronization;
- wireless channel simulation;
- FEC/channel coding;
- manuscript rewriting;
- DGN-CC theoretical rewrite;
- checkpoint changes.

This task is intended to generate evaluation evidence for a later manuscript revision.

---

## 26. Scientific decision after the first run

Compare:

```text
unrestricted
greedy / K=1
K=2
K=4
K=8
...
```

### Case A — meaningful rate–accuracy trade-off

If performance improves materially with K/rate, the no-retraining ICASSP direction is viable. Make the rate–accuracy curve the central empirical result.

### Case B — greedy is nearly as good

Do not conceal it. The shared-policy communication model likely needs a deeper conceptual revision because deterministic shared-policy execution may remove most action communication.

### Case C — small K improves performance

Report it. Top-K may act as inference regularization by suppressing low-probability exploration. Do not force a monotonic narrative.

---

## 27. Definition of done

The task is complete when:

1. Existing unrestricted InfoCost evaluation still runs.
2. A new evaluation-only rate sweep runs from existing checkpoints.
3. Greedy/Top-K modify action choice without modifying weights.
4. Hits@1 is MINERVA-compatible.
5. PED uses the top-ranked predicted path.
6. A budget-vs-Hits@1 curve is produced.
7. A non-uniform task-agnostic coding baseline is produced.
8. Entropy, surprisal, and integer source-code length are separate.
9. Success-conditioned and horizon diagnostics are reported.
10. Action-cap truncation is quantified or explicitly unavailable.
11. All four current configs export to a common CSV/JSON format.
12. No training/checkpoint modification occurred.

---

## 28. Final instruction to Codex

Before editing, read:

```text
code/evaluation_infocost.py
code/policy_entropy/eval.py
code/policy_entropy/metrics.py
code/policy_entropy/artifacts.py
code/policy_entropy/plotting.py
run_infocost.sh
configs/*.yaml
minerva/code/model/trainer.py
minerva/code/model/environment.py
minerva/code/data/grapher.py
minerva/code/options.py
```

Pay special attention to:

- rollout ranking in `TrainerNLQ.test()`;
- top-ranked rollout selection for PED;
- single vs multi-answer behavior in `EpisodeNLQ.get_reward()`;
- gold hop count via `EpisodeNLQ.get_path_length()`;
- `max_num_actions` truncation in the grapher;
- unknown-argument/YAML rejection in `read_options()`.

Prioritize **correctness and comparability with the current evaluator** over refactoring elegance.

If repository behavior conflicts with this brief, preserve current semantics and document the conflict rather than silently redefining the experiment.


---

## 29. Implementation status — updated 2026-08-28

P0 is implemented as an additive, evaluation-only path. No model was retrained, no checkpoint was modified, and no existing YAML configuration, `code/evaluation_infocost.py`, manuscript file, or MINERVA submodule source was changed. The submodule remains at `9bf1ae998d14471c3f7c31f70969d0bbf9873329`; its two manuscript deletions were present before this work.

### Added or modified files

```text
code/evaluation_rate_sweep.py
code/policy_entropy/rate_constraints.py
code/policy_entropy/rate_eval.py
code/policy_entropy/rate_plotting.py
run_rate_sweep.sh
tests/__init__.py
tests/test_rate_constraints.py
tests/test_rate_eval.py
README.md
```

The implementation loads one existing checkpoint and evaluates greedy, Top-K, and upstream TensorFlow-policy modes with `use_beam=False` and `pool="max"`. Top-K rollout ranking uses cumulative execution-policy probability. Hits@1/MRR retain MINERVA duplicate-terminal-entity semantics, and PED uses the highest-scoring rollout. Single-answer and multi-answer reward handling remains dataset-native.

The task-agnostic distribution is a smoothed graph-structural relation prior built from graph actions, not evaluation labels or selected test actions. Entropy, selected-action surprisal, Shannon integer length, fixed-rank budget, task-agnostic cross entropy/KL, correctness-conditioned costs, gold-hop diagnostics, and validated action-cap diagnostics are reported separately.

### Validation completed

- Existing Kinship InfoCost evaluation ran before and after implementation with exactly identical summary statistics.
- The new unrestricted `tf_policy` entropy statistics agree with the existing evaluator within `1e-6`.
- All 15 pure NumPy unit tests passed in the `minerva_tf2` conda environment at the end of P0.
- Python compilation, shell syntax checks, and `git diff --check` passed.
- Greedy and Top-K with `K=1` are identical and report zero execution payload.
- Checkpoint file sizes and modification times are unchanged.
- Full-dataset sweeps completed for:
  - `configs/kinshiphinton.yaml`
  - `configs/metaqa.yaml`
  - `configs/mquake_st_single.yaml`
  - `configs/mquake_st_multi.yaml`
- All current configs use `test_rollouts: 100`; this is confirmed both in the YAMLs and in `rate_eval.py`, which calls `change_test_rollouts(trainer.test_rollouts)` and constructs every evaluation episode with `trainer.test_rollouts`.

### Full-sweep results available so far

All values below are from evaluation seed 42, 100 rollouts/question, `use_beam=False`, and MINERVA-compatible `pool="max"` ranking.

| Dataset | Greedy Hits@1 | Unrestricted Hits@1 | Greedy MRR | Unrestricted MRR | Greedy path metric | Unrestricted path metric | Unrestricted policy entropy (bits/hop) | Truncated visited states |
|---|---:|---:|---:|---:|---|---|---:|---:|
| Kinship | 0.960396 | 0.960396 | 0.960396 | 0.976073 | PED 1.861386 / RED 1.6436 | PED 1.990099 / RED 1.7129 | 1.0746 | 0.00% |
| MetaQA | 0.897910 | 0.897757 | 0.897910 | 0.930447 | unavailable | unavailable | 0.4540 | 1.92% |
| MQuAKE-ST Single | 0.8863 | 0.8863 | 0.8863 | 0.8889 | PED 0.9215 / RED 0.6602 | PED 0.9215 / RED 0.6602 | 0.0066 | 5.15% |
| MQuAKE-ST Multi | 0.8563 | 0.8563 | 0.8563 | 0.8602 | RED 1.2023 | RED 1.2023 | 0.0110 | 18.39% |

MQuAKE-ST Multi does not have a compatible entity-path PED under the current data representation; keep RED separate rather than relabeling it PED.

### MetaQA Top-K detail

MetaQA has 39,093 questions and 100 rollouts/question.

| Mode | Mean fixed local-rank bits/hop | Hits@1 | MRR |
|---|---:|---:|---:|
| Greedy / K=1 | 0.000 | 0.897910 | 0.897910 |
| K=2 | 1.000 | 0.897782 | 0.928080 |
| K=4 | 1.896 | 0.897859 | 0.929972 |
| K=8 | 2.683 | 0.897757 | 0.930104 |
| K=16 | 3.210 | 0.897757 | 0.930133 |
| K=32 | 3.404 | 0.897731 | 0.930122 |
| K=64 | 3.469 | 0.897782 | 0.930141 |
| K=128 | 3.498 | 0.897757 | 0.930128 |
| Unrestricted TF policy | 3.519 | 0.897757 | 0.930447 |

MetaQA's unrestricted graph/action diagnostics:

```text
mean policy entropy:              0.45395 bits/hop
mean task-agnostic surprisal:     3.50262 bits/hop
policy-vs-task-agnostic KL:       3.04868 bits/hop
max raw valid actions:            4305
overall visited-state truncation: 1.9165%
hop-2 truncation:                 ~5.70%
```

### Current scientific observation

Do **not** force a monotonic rate–accuracy interpretation.

The observed result is:

1. Greedy/K=1 preserves unrestricted Hits@1 on all four completed settings under the present 100-rollout MINERVA evaluation.
2. MQuAKE-ST Single/Multi are effectively deterministic under the pretrained policy, so increasing Top-K support produces almost no task-metric change.
3. Kinship has small non-monotonic K=2/K=4 changes that are single-seed observations and should not be interpreted as established improvements.
4. MetaQA has essentially flat Hits@1 but a large difference between greedy and stochastic MRR.
5. The task-agnostic structural prior is much less predictive than the task-conditioned policy, especially on MQuAKE and MetaQA.
6. Action-cap truncation is a real limitation, especially MQuAKE-ST Multi at 18.39% of visited states.

---

## 30. Important correction — 100-rollout ranking utility is not a per-rollout communication task

The current implementation is correctly using 100 evaluation rollouts/question.

Upstream MINERVA Hits@1/MRR under `pool="max"` are **candidate-ranking metrics built from the set of sampled rollouts**:

1. sample `R=test_rollouts` trajectories;
2. score each trajectory by cumulative execution-policy log probability;
3. rank those trajectories;
4. collapse duplicate incorrect terminal entities;
5. compute the rank of the first correct terminal entity.

Therefore the current MetaQA MRR is valid as **MINERVA 100-rollout candidate-ranking MRR**. It is not conventional full-KG entity-ranking MRR.

However, it is misleading to compare that ensemble MRR directly against only:

```text
1 bit/hop
```

when `K=2`, because `1 bit/hop` is a **per-rollout action payload**.

For a fixed 3-hop horizon and 100 independently communicated rollouts, the worst-case hard local-rank action payload for K=2 can be approximately:

```text
100 rollouts * 3 hops * 1 bit = 300 action bits/question
```

subject to the actual per-state support size.

### Consequence

The next evaluator revision must explicitly separate:

```text
A. single-trajectory execution
   utility: terminal success / path fidelity
   communication: bits for one executed path

B. multi-rollout candidate ranking
   utility: MINERVA Hits@1 / MRR
   communication: total bits required for the complete rollout ensemble
```

Do not describe an MRR improvement as being obtained for only `B bits/hop` without also reporting the rollout count and the corresponding total per-question ensemble communication.

### Greedy special case

With deterministic greedy action selection and identical initial state/task, all repeated rollout slots for one question follow the same trajectory.

Therefore for greedy:

```text
MRR == Hits@1
```

under the current MAX-pool ranking semantics.

This is expected, not an evaluator bug.

---

## 31. Immediate follow-up implementation plan

> **Status: COMPLETE.** This section preserves the historical follow-up plan that was subsequently implemented and evaluated. See Sections 33–36 and `codex_ref/infocost_experiment_results_20260828.md` for the final status and results.

The planned work was evaluation-only and additive.

### Priority 1 — add an explicit evaluation-rollout override

Add a rate-only CLI argument such as:

```text
--rate_test_rollouts
```

or:

```text
--rate_eval_rollouts
```

Requirements:

1. default `None` means preserve the YAML `test_rollouts`;
2. consume it in the custom pre-parser before upstream `read_options()`;
3. do not modify existing YAML files;
4. when set, apply it before constructing `TrainerNLQ`;
5. save both:
   - configured/YAML rollout count;
   - effective evaluation rollout count;
6. preserve current behavior when omitted.

This is needed to run:

```text
R=1
```

as a true single-trajectory protocol without changing the checkpoint.

#### Single-trajectory outputs

For `R=1`, report at minimum:

```text
single_rollout_success_rate
hits_at_1                  # should numerically equal success under one candidate
PED / RED where available
mean_path_fixed_rank_bits
mean_path_surprisal_bits
mean_path_shannon_code_bits
```

Do not make MRR a central single-trajectory metric; with one candidate it collapses to the same binary outcome as Hits@1.

### Priority 2 — add total per-question communication for multi-rollout evaluation

The current fields average over rollout/hop positions. Keep them for diagnostics.

Also compute **per-question total communication across all rollouts**.

Given a tensor:

```text
cost_bits shape = [B, R, T]
```

compute:

```text
question_total_cost_bits[b] = sum_{r=1..R} sum_{t=1..T} cost_bits[b,r,t]
```

Add summary fields such as:

```text
mean_question_fixed_rank_bits
mean_question_surprisal_bits
mean_question_shannon_code_bits
mean_question_entropy_sum_bits
```

If task-agnostic fields are available, also add:

```text
mean_question_task_agnostic_surprisal_bits
mean_question_task_agnostic_shannon_bits
```

These are ensemble costs for the currently sampled rollout set.

Do **not** replace the existing per-hop/per-rollout means. Save both.

Metadata should state:

```text
For R>1, MINERVA Hits@1/MRR are rollout-ensemble metrics.
mean_question_* fields sum communication across all R rollout trajectories.
```

#### Plotting rule

For the standard `R=100` MINERVA evaluation:

- if plotting MRR against communication, use a total per-question ensemble cost on the x-axis;
- do not plot MRR against only per-rollout `mean_fixed_budget_bits` without a clear secondary normalization label.

For the `R=1` evaluation:

- plot single-rollout success / path metric against per-path communication.

### Priority 3 — add a NumPy unrestricted-policy mode

Current stochastic comparison has:

```text
Top-K:       NumPy PCG64
unrestricted: TensorFlow categorical
```

This introduces a finite-sample backend mismatch.

Add a mode such as:

```text
action_mode = "numpy_policy"
```

or:

```text
action_mode = "unrestricted_numpy"
```

Behavior:

1. fetch the same original `trainer.test_logits`;
2. use the complete valid action support;
3. normalize over valid actions in NumPy;
4. sample using the same `np.random.default_rng(rate_seed)` path as Top-K;
5. use those original-policy log probabilities as the execution-policy scores;
6. override the action/relation exactly as Top-K does.

Keep:

```text
tf_policy
```

as the regression reference. Do not remove it.

#### Matched-RNG requirement

When Top-K and `numpy_policy` are run with the same seed, initialize their RNG streams identically.

This provides common underlying uniform draws where the trajectories/states remain comparable. Once trajectories diverge, they are not fully paired; document this rather than claiming exact paired sampling.

#### Required test

On a synthetic batch where:

```text
K >= valid_action_count
```

Top-K and `numpy_policy`, given identical log probabilities and identical RNG state, must select the same actions samplewise.

### Priority 4 — repeat the most informative stochastic experiments across seeds

After `numpy_policy` exists, run at least:

```text
seeds = 5 values
```

for:

```text
Kinship
MetaQA
```

with:

```text
R=100
modes = greedy, K=2, K=4, numpy_policy
```

The purpose is:

- determine whether Kinship's one-question K=2/K=4 Hits@1 change persists;
- quantify MetaQA's MRR variation;
- avoid relying on TensorFlow-vs-NumPy backend differences.

MQuAKE is much closer to deterministic; extensive multi-seed sweeps are lower priority unless the cap-sensitivity experiment changes its behavior.

Do not add a complicated in-process multi-seed framework if it risks TensorFlow reproducibility. It is acceptable to run separate seeded processes and aggregate their exported CSV/JSON files afterward.

### Priority 5 — action-cap sensitivity without retraining

The current cap diagnostics are:

```text
Kinship:          max raw 8,    cap 100, truncated 0.00%
MetaQA:           max raw 4305, cap 200, truncated ~1.92%
MQuAKE-ST Single: max raw 479,  cap 200, truncated ~5.15%
MQuAKE-ST Multi:  max raw 479,  cap 200, truncated ~18.39%
```

MQuAKE-ST Multi needs a sensitivity check.

Add an optional **evaluation-only** cap override, for example:

```text
--rate_max_num_actions_override 512
```

Requirements:

1. default `None` preserves the config;
2. do not edit YAML;
3. apply the override before building the trainer/environment;
4. record configured cap and effective cap in metadata;
5. do not modify checkpoint files;
6. before relying on it, inspect whether any trainable checkpoint tensor shape depends on `max_num_actions`;
7. if checkpoint restore or model semantics depend on the trained cap, stop and report the incompatibility rather than forcing a run.

If safe, run MQuAKE-ST Single and Multi with:

```text
max_num_actions = 512
```

because their observed maximum raw count is 479.

At minimum compare:

```text
greedy
K=2
K=4
numpy_policy
```

at cap 200 vs 512.

The purpose is to test whether the near-deterministic/flat result is an artifact of action truncation.

MetaQA's maximum raw degree is 4305, so 512 would not eliminate all truncation. MetaQA cap sensitivity is secondary because the observed visited-state truncation rate is much smaller.

---

## 32. Tests and definition of done for the follow-up

### New tests

Add tests for:

#### Evaluation-rollout override

- omitted override preserves YAML `test_rollouts`;
- override `R=1` creates one rollout/question;
- output metadata records configured and effective rollout counts.

#### Total question communication

For synthetic `[B,R,T]` arrays:

- question total is exactly `sum(axis=(1,2))`;
- mean question total is the mean over B;
- greedy deterministic support yields zero hard rank / Shannon execution cost;
- for nonnegative costs, total question cost is at least the corresponding single-rollout/path cost.

#### NumPy unrestricted mode

- invalid actions are impossible;
- distribution equals the original policy normalized over valid actions;
- with identical RNG state and `K >= valid_count`, Top-K and NumPy unrestricted select identical actions;
- execution-policy cumulative score uses the same NumPy unrestricted distribution.

#### Protocol semantics

- with `R=1`, Hits@1 equals rollout success;
- greedy with repeated identical rollout slots has `MRR == Hits@1`;
- `num_rollouts` is always emitted;
- multi-rollout MRR outputs also emit total per-question communication fields.

#### Cap override

- default preserves configured cap;
- override is metadata-visible;
- no checkpoint path/weight modification occurs;
- fail clearly if checkpoint/model construction is incompatible.

### Required regression checks

Before accepting the follow-up:

1. Existing `evaluation_infocost.py` remains unchanged in behavior.
2. Existing P0 `tf_policy` mode remains numerically consistent with the previous output for the same config/seed.
3. Existing `R=100` outputs remain available and retain their definitions.
4. `numpy_policy` and `tf_policy` have statistically compatible unrestricted distributions; exact samplewise equality is not required because they use different RNG engines.
5. No model retraining or checkpoint modification occurs.
6. MINERVA submodule remains unmodified unless a blocking issue is explicitly documented first.

### Immediate experimental matrix after implementation

Run the smallest informative matrix first:

```text
Protocol A: single trajectory
R = 1
modes = greedy, K=2, K=4, numpy_policy
datasets = Kinship, MetaQA, MQuAKE-ST Single, MQuAKE-ST Multi

Protocol B: standard MINERVA ranking
R = 100
modes = greedy, K=2, K=4, numpy_policy
datasets = Kinship, MetaQA
seeds = at least 5

Protocol C: cap sensitivity
R = 1 and/or 100 as compute permits
datasets = MQuAKE-ST Single, MQuAKE-ST Multi
caps = 200, 512
modes = greedy, K=2, K=4, numpy_policy
```

Do not immediately rerun every K in `{1,2,4,8,16,32,64,128}` across every seed. The existing full sweeps already show saturation; the compact matrix above should answer the remaining scientific questions with less compute.

### Follow-up definition of done

The follow-up is complete when:

1. single-trajectory (`R=1`) execution can be run without changing YAML/checkpoints;
2. total per-question communication is available for `R>1` ensemble ranking;
3. MRR can be paired with ensemble-level communication rather than only per-rollout bits/hop;
4. NumPy unrestricted sampling provides a backend-matched stochastic reference for Top-K;
5. Kinship/MetaQA stochastic comparisons have multiple-seed evidence;
6. MQuAKE action-cap sensitivity is either measured at 512 or explicitly shown to be incompatible with the checkpoint architecture;
7. all new outputs retain the distinction between:
   - per-hop rate;
   - per-path cost;
   - per-question ensemble cost;
   - task utility;
8. no retraining/checkpoint mutation occurs.

### Do not do yet

> **Historical condition, now superseded:** The restrictions below applied while the follow-up was unfinished. The follow-up definition of done has now been satisfied; current guidance is in Sections 33–36.

Until the above was complete:

- do not rewrite the manuscript around MetaQA MRR;
- do not claim `K=2` gives the MRR gain for only `1 bit/hop` without ensemble-cost qualification;
- do not call the observed flat Hits@1 curve a universal zero-communication theorem;
- do not remove the task-agnostic baseline;
- do not hide the MQuAKE action-cap truncation;
- do not add communication-aware training.

The purpose of this follow-up is to make the **units of communication and utility operationally consistent** before the 4-page ICASSP manuscript is finalized.

---

## 33. Follow-up implementation completion — 2026-08-28

The evaluation-only follow-up is complete:

- an explicit evaluation-rollout override supports `R=1` without YAML changes;
- total per-question communication sums costs over all `R` rollouts and `T` hops;
- backend-matched `numpy_policy` sampling is available while `tf_policy` remains the regression reference;
- an evaluation-only `max_num_actions` override supports checkpoint-compatible cap sensitivity;
- all 22 pure-Python `unittest` tests passed in the `minerva_tf2` Conda environment;
- Python compilation, shell syntax, and `git diff --check` checks passed;
- the existing P0 `tf_policy` output remained numerically unchanged in its regression comparison;
- no checkpoint, YAML configuration, dataset, manuscript, or MINERVA submodule source was modified;
- checkpoint identities remained unchanged during evaluation.

No retraining was performed. All Python execution, validation, and experiment commands used `minerva_tf2`.

## 34. Completed experiment artifact index

### Original full `R=100` rate sweeps

| Dataset | Output directory |
|---|---|
| Kinship | `saved_models/kinshiphinton/20260827_191947` |
| MQuAKE-ST Single | `saved_models/mquake_st/20260827_193842` |
| MQuAKE-ST Multi | `saved_models/mquake_st/20260827_194422` |
| MetaQA | `saved_models/metaqa/20260827_194812` |

### Stage 1 — `R=1` single-trajectory evaluation

| Dataset | Output directory |
|---|---|
| Kinship | `saved_models/kinshiphinton/20260828_015559` |
| MQuAKE-ST Single | `saved_models/mquake_st/20260828_020226` |
| MQuAKE-ST Multi | `saved_models/mquake_st/20260828_020333` |
| MetaQA | `saved_models/metaqa/20260828_020631` |

### Stage 2 — MQuAKE evaluation-time cap 512 sensitivity

| Dataset | R | Output directory |
|---|---:|---|
| MQuAKE-ST Single | 1 | `saved_models/mquake_st/20260828_021451` |
| MQuAKE-ST Multi | 1 | `saved_models/mquake_st/20260828_021656` |
| MQuAKE-ST Single | 100 | `saved_models/mquake_st/20260828_021815` |
| MQuAKE-ST Multi | 100 | `saved_models/mquake_st/20260828_022724` |

### Stage 3A — Kinship `R=100` multi-seed

| Rate seed | Output directory |
|---:|---|
| 42 | `saved_models/kinshiphinton/20260828_021222` |
| 43 | `saved_models/kinshiphinton/20260828_021248` |
| 44 | `saved_models/kinshiphinton/20260828_021313` |
| 45 | `saved_models/kinshiphinton/20260828_021339` |
| 46 | `saved_models/kinshiphinton/20260828_021405` |

### Stage 3B — MetaQA `R=100` multi-seed

| Rate seed | Output directory |
|---:|---|
| 42 | `saved_models/metaqa/20260828_030323` |
| 43 | `saved_models/metaqa/20260828_040427` |
| 44 | `saved_models/metaqa/20260828_050525` |
| 45 | `saved_models/metaqa/20260828_060604` |
| 46 | `saved_models/metaqa/20260828_070659` |

Every listed CSV, JSON, and metadata artifact was revalidated on 2026-08-28. Dataset, question count, configured/effective rollout count and cap, rate seed, modes, `use_beam=False`, `pool="max"`, checkpoint identity, raw-action maximum, and truncation diagnostics agree with the intended protocol. Full tables are in `codex_ref/infocost_experiment_results_20260828.md`.

## 35. Final experimental conclusions

1. `R=1` and `R=100` measure different operational utilities. For `R=1`, Hits@1 equals executed-trajectory success and communication is a per-path cost. For `R=100`, Hits@1/MRR are MINERVA sampled-rollout candidate-ranking metrics and communication must be totaled over the complete ensemble.
2. Stochastic `R=1` execution provides no general success advantage. Kinship K=2 is the important exception: it improves from 97/101 to 99/101 at seed 42, while worsening PED/RED. MQuAKE and MetaQA stochastic modes slightly underperform greedy.
3. `R=100` MRR is not conventional full-KG entity-ranking MRR. It ranks candidates sampled by the 100-rollout ensemble under MINERVA `pool="max"` semantics.
4. Across five MetaQA rate seeds, greedy/K=2/K=4/NumPy Hits@1 are approximately 0.897910/0.897782/0.897859/0.897792—only a few questions apart among 39,093—while stochastic MRR improves by about 0.030–0.032. K=2 recovers 93.103% ± 0.344% of the NumPy MRR gain and K=4 recovers 99.486% ± 0.062% (population SD). Total fixed-rank costs for the 100-rollout, three-hop ensemble are about 300, 568.8, and 1055.7 bits/question for K=2, K=4, and NumPy.
5. Across five Kinship rate seeds, NumPy Hits@1 matches greedy in every seed. K=2/K=4 gain exactly one question in four of five seeds; this is not enough to establish inference regularization. Stochastic modes improve ensemble MRR but have worse PED/RED than greedy.
6. MQuAKE cap 512 removes observed truncation because the maximum raw action count is 479. Single changes only slightly; Multi is materially cap-sensitive in its absolute baseline. Removing truncation does not reveal a stochastic-navigation advantage, and the policy remains near-deterministic. Cap 512 is only an evaluation sensitivity analysis because training used cap 200.
7. The task-agnostic graph-structural prior is substantially more expensive than the task-conditioned policy.
8. The evidence does not support a monotonic communication-versus-accuracy law.
9. The strongest operational distinction is that communication needed for single/top-1 navigation utility is not the same as communication useful for ranked sampled-candidate utility.
10. Greedy zero action payload is conditional on synchronized policy, task/state, candidate action set, and action ordering/tie-breaking. It is not a universal zero-communication theorem.

## 36. Current project status

- **Core no-retraining implementation:** complete.
- **Core no-retraining experiments:** complete.
- **Additional core experiments required:** no.
- **Next phase:** ICASSP manuscript revision.
- **Automatic experiment launch:** prohibited unless separately requested.
- **Optional diagnostics:** explicit unique-terminal-candidate diversity could be informative but is not required for the current conclusions.
- **Canonical quantitative tracker:** `codex_ref/infocost_experiment_results_20260828.md`.
