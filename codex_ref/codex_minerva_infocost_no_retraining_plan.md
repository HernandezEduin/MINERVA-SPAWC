# Codex Implementation Brief — MINERVA-InfoCost No-Retraining Revision

**Target repository:** `https://github.com/HernandezEduin/MINERVA-InfoCost`  
**Default branch:** `master`  
**Pinned MINERVA submodule:** `9bf1ae998d14471c3f7c31f70969d0bbf9873329`  
**Prepared:** 2026-08-27  
**Updated:** 2026-08-29 after independent ICASSP manuscript review  
**Scope:** Evaluation-only changes. **Do not retrain MINERVA. Do not modify pretrained checkpoints.**

---

## 1. Current status and objective

The original no-retraining rate-sweep implementation and follow-up experiments are complete. The evaluator already supports:

- deterministic greedy execution;
- stochastic Top-K execution;
- original TensorFlow unrestricted sampling (`tf_policy`) as a regression reference;
- backend-matched NumPy unrestricted sampling (`numpy_policy`);
- explicit evaluation rollout overrides, including true `R=1` execution;
- total per-question communication over `R>1` rollout ensembles;
- MINERVA-compatible `pool="max"` Hits@1/MRR;
- path/relation edit-distance diagnostics where available;
- task-agnostic structural priors and source-coding diagnostics;
- evaluation-only `max_num_actions` overrides;
- action-cap truncation diagnostics.

The completed quantitative record is tracked in:

```text
codex_ref/infocost_experiment_results_20260828.md
```

The previous status statement that **no further core experiment is required is now superseded**.

Independent manuscript review identified one potentially outcome-changing scientific question:

> **Can a deterministic diverse decoder recover the multi-rollout ranking benefit without stochastic action-realization messages?**

Under the manuscript's synchronized shared-policy assumptions, both endpoints can in principle execute the same deterministic beam/top-path procedure. If so, candidate diversity may not require stochastic action communication.

The new post-review goal is therefore to compare communicated stochastic branching against a deterministic zero-stochastic-message diverse decoder, while directly measuring candidate diversity and answer coverage.

---

## 2. Hard constraints

- **No retraining.**
- **Do not change checkpoint weights.**
- **Do not add communication-aware loss terms.**
- **Do not regenerate or change datasets.**
- **Do not change the existing YAML configs unless absolutely unavoidable.**
- **Do not modify the `minerva/` submodule unless a blocking incompatibility is demonstrated first.**
- Preserve all existing rate-evaluator behavior when new functionality is disabled.
- Preserve `tf_policy` as the historical TensorFlow regression reference.
- Use `numpy_policy` as the preferred scientific unrestricted stochastic reference for new matched comparisons.
- Do not silently change MINERVA Hits@1/MRR semantics.
- Do not reinterpret Top-K fixed-rank payload as an entropy/source-coding limit.
- Do not call deterministic beam zero-total-communication; only its **incremental stochastic action-realization payload** is zero under the shared-side-information accounting.
- Do not launch the full expensive experiment campaign automatically. Implement, test, smoke-test, and prepare CPU launch scripts for the user.
- Do not modify manuscript files in this repository task.
- Do not commit or push automatically unless explicitly instructed by the user.

### Environment

All Python commands/tests/smoke evaluations must use:

```text
minerva_tf2
```

Prefer:

```bash
conda run -n minerva_tf2 python ...
conda run -n minerva_tf2 bash ...
```

The user is running CPU-only. Do not assume a GPU is available.

---

## 3. Current repository state to inspect first

Before editing, inspect the current repository rather than relying on this document alone:

```bash
git status
git rev-parse HEAD
git submodule status
```

Read in full:

```text
codex_ref/codex_minerva_infocost_no_retraining_plan.md
codex_ref/infocost_experiment_results_20260828.md
code/evaluation_rate_sweep.py
code/policy_entropy/rate_constraints.py
code/policy_entropy/rate_eval.py
code/policy_entropy/rate_plotting.py
run_rate_sweep.sh
tests/
minerva/code/model/trainer.py
minerva/code/model/environment.py
minerva/code/data/grapher.py
minerva/code/options.py
```

The current rate evaluator intentionally uses `use_beam=False` for its existing greedy/Top-K/stochastic modes. The new deterministic-diverse baseline is the explicit exception being investigated in this follow-up.

---

## 4. Existing evaluation semantics that must remain unchanged

### 4.1 `R=1`

`R=1` is a true one-rollout-per-question protocol.

For `R=1`:

```text
Hits@1 == single executed-trajectory success
```

MRR adds no information beyond the binary terminal outcome and should not be a headline `R=1` metric.

Communication is naturally interpreted per executed path.

### 4.2 `R=100`

For `R=100`, MINERVA's `pool="max"` evaluation is a sampled-candidate ranking protocol.

The implemented evaluator reproduces upstream semantics:

1. accumulate each rollout's cumulative execution-policy log probability;
2. reshape to `[num_questions, R]`;
3. sort rollouts by descending cumulative score;
4. collapse duplicate incorrect terminal entities when counting rank;
5. Hits@1 is true when the first correct sampled terminal entity has rank 1;
6. MRR is reciprocal rank of the first correct sampled terminal entity, or zero when no correct entity is sampled.

This MRR is **not conventional full-KG entity-ranking MRR**.

For `R=100`, communication plotted against Hits@1/MRR must use ensemble-level per-question communication, not only a per-hop/per-rollout rate.

### 4.3 Greedy

With synchronized policy/state/action support/order/tie-breaking, greedy argmax can be reproduced at both endpoints:

```text
incremental stochastic action-realization payload = 0
```

Repeated greedy rollout slots follow the same deterministic trajectory, so under current MAX-pool semantics:

```text
MRR == Hits@1
```

This is expected.

### 4.4 Top-K

Top-K retains the `K_t=min(K,|A_t|)` highest-probability valid actions, renormalizes, and samples from the truncated execution policy.

The hard fixed-rank identifier cost is:

```text
ceil(log2(K_t))
```

This is a protocol-specific local-rank payload, not an entropy/source-coding minimum.

For Top-K ranking, cumulative path score must use the truncated/renormalized **execution policy**, not the original untruncated policy.

### 4.5 NumPy unrestricted

`numpy_policy` samples from the full valid policy support with the same NumPy PCG64 sampling path used by Top-K.

Use it as the preferred unrestricted scientific reference for new comparisons.

Keep `tf_policy` only as the historical upstream regression reference.

---

## 5. Completed no-retraining evidence

Detailed artifacts and exact values are in `codex_ref/infocost_experiment_results_20260828.md`.

High-level established findings:

1. `R=1` and `R=100` measure different operational utilities.
2. Stochastic `R=1` execution gives no consistent single-trajectory success advantage.
3. MetaQA `R=100` has nearly unchanged Hits@1 but a substantial sampled-candidate MRR gain under stochastic execution.
4. Top-2 recovers about 93.1% of the backend-matched unrestricted MetaQA MRR gain at about 300 fixed-rank action bits/question for the 100-rollout, three-hop ensemble.
5. Top-4 recovers about 99.5% at about 568.8 fixed-rank bits/question.
6. Kinship shows a smaller multi-rollout MRR effect.
7. MQuAKE-ST Single/Multi are near-deterministic under the trained policy and show only small `R=100` MRR gains.
8. MQuAKE-ST Multi is sensitive in absolute performance to the cap-200 truncation, but cap 512 does not reveal a stochastic top-1 advantage.
9. There is no evidence for a universal monotonic communication-versus-accuracy law.

The manuscript now uses cross-dataset `R=100` evidence, so the new follow-up must preserve cross-dataset comparability.

---

## 6. Post-review Priority 1 — deterministic zero-stochastic-message diverse baseline

This is the highest-priority open experiment.

### Scientific question

Current stochastic `R=100` modes gain MRR by producing multiple sampled trajectories. A reviewer can reasonably ask:

> Could both endpoints instead run the same deterministic beam/top-path decoder and obtain candidate diversity with zero stochastic action-realization messages?

If yes, the observed benefit may be primarily the value of **candidate diversity**, not stochastic communication itself.

### First inspect existing beam search

Pinned upstream `TrainerNLQ.test(..., beam=True)` already contains deterministic beam-search logic.

Before implementing anything new, verify from the actual code that it:

- maintains top cumulative-probability paths;
- uses deterministic score-based selection;
- sets `effective_rollouts = min(test_rollouts, max_num_actions)`;
- evaluates resulting candidates with MINERVA-compatible ranking semantics.

For the current configs, `R=100` and action caps >=100 should permit a 100-path deterministic beam baseline.

### Preferred implementation strategy

Use the smallest safe approach:

1. **Preferred:** reuse upstream beam logic externally if it exposes enough outputs.
2. If upstream `test()` cannot expose terminal candidates/scores needed for the new diagnostics, reproduce its pinned beam semantics in evaluation-only top-level code.
3. Do **not** edit the submodule unless there is no safe alternative.

If beam logic is mirrored externally, add a regression showing that beam Hits@1/MRR match upstream `TrainerNLQ.test(..., beam=True)` under the same config/width.

### Main deterministic mode

Add a scientific mode such as:

```text
action_mode = "deterministic_beam"
```

or a separate beam-evaluation entry point if cleaner.

Main settings:

```text
beam width = 100
R = 100 candidate/path budget
fixed pretrained checkpoint
same configured action cap
```

### Communication interpretation

Deterministic beam performs no stochastic sample realization.

Under the same synchronized shared-policy/state/local-interface/order/tie-breaking assumptions:

```text
incremental stochastic action-realization payload = 0
```

This does **not** mean:

- zero compute;
- zero synchronization cost;
- zero state/interface transport;
- zero total communication.

Do not assign a fake fixed-rank/entropy cost to deterministic beam branches. Use explicit null/NA for stochastic-message cost fields where appropriate and a separate metadata field explaining the zero incremental stochastic-realization interpretation.

### Matching rule

The main comparison should match the logical candidate/path budget as closely as possible:

```text
deterministic beam width 100
vs
stochastic R=100
```

Do not claim perfectly matched compute unless it is actually measured and equal. Beam has a different computational search cost.

---

## 7. Post-review Priority 2 — candidate diversity and answer-coverage diagnostics

The current manuscript interpretation says stochastic branching can expose additional candidate trajectories, but this mechanism has not yet been measured directly.

Add inexpensive `R>1` diagnostics for:

```text
greedy
topk
numpy_policy
deterministic_beam
```

### Required per-question quantities

#### 7.1 Unique terminal candidates

```text
unique_terminal_candidate_count
```

Number of distinct terminal entity IDs among the logical trajectories/candidates.

#### 7.2 Unique terminal fraction

```text
unique_terminal_fraction = unique_terminal_candidate_count / candidate_count
```

For stochastic modes, `candidate_count=R`.
For deterministic beam, use the actual number of retained final beams/candidates and record that count explicitly.

#### 7.3 Candidate answer coverage

```text
candidate_answer_coverage
```

Boolean per question: at least one candidate terminal entity is a valid answer.

Do **not** call this Hits@K.

#### 7.4 MRR conditional on answer coverage

```text
mrr_given_answer_coverage
```

Use the existing sampled-candidate reciprocal rank but average only over questions where `candidate_answer_coverage=True`.

### Required aggregate summaries

At minimum export:

```text
mean_unique_terminal_candidates
median_unique_terminal_candidates
mean_unique_terminal_fraction
candidate_answer_coverage
mrr_given_answer_coverage
```

Optional if essentially free:

```text
mean_unique_correct_terminal_candidates
mean_correct_rollouts_per_question
```

### Questions these diagnostics must answer

- Does Top-2 produce substantially more unique candidates than greedy?
- Does deterministic beam produce more or fewer unique candidates than Top-2/Top-4?
- Does MetaQA's MRR improvement mainly come from better answer coverage?
- Conditional on a correct answer being present, which decoder ranks it best?
- Do MQuAKE's near-deterministic policies simply fail to create candidate diversity?

---

## 8. Post-review Priority 3 — uniform NumPy unrestricted cross-dataset reference

The current cross-dataset manuscript table combines historical results from different unrestricted sampling backends.

For the new post-review campaign, use `numpy_policy` as the uniform unrestricted stochastic reference on **all four datasets**.

Do not delete or overwrite historical TensorFlow results.

The new seed-42 `R=100` matrix must contain:

```text
greedy
Top-2
Top-4
numpy_policy
deterministic_beam
```

for:

```text
configs/kinshiphinton.yaml
configs/metaqa.yaml
configs/mquake_st_single.yaml
configs/mquake_st_multi.yaml
```

This automatically supplies the currently missing cap-200 NumPy unrestricted `R=100` values for both MQuAKE-ST splits.

---

## 9. Primary post-review experiment matrix

### Protocol

```text
R = 100
rate seed = 42
CPU-only
```

### Datasets and configured caps

| Dataset | Config | Cap | Horizon |
|---|---|---:|---:|
| Kinship | `configs/kinshiphinton.yaml` | 100 | 3 |
| MetaQA | `configs/metaqa.yaml` | 200 | 3 |
| MQuAKE-ST Single | `configs/mquake_st_single.yaml` | 200 | 4 |
| MQuAKE-ST Multi | `configs/mquake_st_multi.yaml` | 200 | 4 |

### Modes

```text
greedy
Top-2
Top-4
NumPy unrestricted
deterministic beam width 100
```

### Required outputs

- standard MINERVA-compatible Hits@1;
- sampled-candidate MRR;
- rollout success rate where meaningful;
- total fixed-rank action payload/question for stochastic Top-K modes;
- unrestricted local-rank reference for `numpy_policy` where currently defined;
- deterministic-beam stochastic action-realization payload = 0 under the conditional accounting;
- all candidate-diversity/coverage diagnostics from Section 7;
- configured/effective rollout count;
- configured/effective action cap;
- truncation diagnostics;
- checkpoint identity;
- root and submodule git SHAs.

---

## 10. MQuAKE cap-512 sensitivity for the new decoder

MQuAKE-ST has raw maximum action count 479, so evaluation cap 512 removes observed truncation.

Prepare a second CPU launcher for:

```text
MQuAKE-ST Single
MQuAKE-ST Multi
R = 100
beam width = 100
max_num_actions override = 512
```

At minimum evaluate:

```text
deterministic beam
```

Preferably, if the launcher/integration makes it straightforward, run the full new comparison at cap 512:

```text
greedy
Top-2
Top-4
numpy_policy
deterministic_beam
```

This is an **evaluation sensitivity** only. Training used cap 200.

Do not silently replace the cap-200 primary results with cap-512 values.

---

## 11. Keep these concepts separate

### Rollout count `R`

Number of logical stochastic trajectories in the existing evaluation protocol.

### Beam width

Number of deterministic paths retained by beam decoding.

### Top-K

Per-state stochastic retained action support.

### Fixed-rank action payload

Protocol-specific bits needed to identify a sampled action within the shared retained support/order.

### Deterministic beam

No stochastic realization is sampled. Under the conditional paper accounting its incremental stochastic action-message payload is zero, but compute and synchronization costs remain.

Do not merge these quantities into one generic "rate" variable.

---

## 12. Required tests

Use the existing `unittest` style. Do not install pytest merely for this task.

Run:

```bash
conda run -n minerva_tf2 python -m unittest discover -s tests -v
conda run -n minerva_tf2 python -m compileall code tests
bash -n run_rate_sweep.sh
git diff --check
```

### 12.1 Candidate diagnostics

Synthetic tests must verify:

- duplicate terminal IDs are counted once for unique-candidate metrics;
- unique count/fraction are correct;
- candidate answer coverage is correct;
- `mrr_given_answer_coverage` excludes uncovered questions rather than treating them as zero;
- existing Hits@1/MRR outputs remain unchanged.

### 12.2 Greedy diagnostic

For deterministic greedy with repeated `R=100` rollout slots:

```text
unique terminal candidates/question = 1
candidate answer coverage = greedy terminal success/Hits@1
MRR == Hits@1
```

assuming no external state nondeterminism.

### 12.3 Beam determinism

- repeated deterministic-beam execution must not depend on the rate sampling seed;
- no NumPy stochastic sampler should drive beam selection.

### 12.4 Beam-width-one sanity

Where the upstream semantics align, beam width 1 should reproduce greedy trajectory/result behavior.

### 12.5 Upstream beam regression

If external beam logic is implemented, compare it against pinned upstream `TrainerNLQ.test(..., beam=True)` on a small controlled run.

Hits@1/MRR should match for the same beam width/config unless a documented upstream limitation prevents exact matching.

### 12.6 Existing regressions

- all existing tests must remain passing;
- current `tf_policy` regression output must remain unchanged;
- existing Top-K and `numpy_policy` sampling behavior must remain unchanged;
- checkpoint files must remain unchanged.

---

## 13. Smoke tests only before user launch

After implementation, Codex may run only inexpensive smoke tests, preferably:

```text
Kinship, one batch, R=100:
greedy / Top-2 / Top-4 / NumPy / deterministic beam-100
```

Optionally one MetaQA batch if needed for validation.

Verify:

- deterministic beam executes successfully;
- candidate diagnostics are populated;
- beam stochastic-action payload is represented as zero/NA consistently and explained in metadata;
- existing modes retain previous metrics on unchanged smoke inputs;
- checkpoint identity remains unchanged.

Do **not** launch full MetaQA automatically.

---

## 14. CPU launchers to create

Create easy-to-run scripts from repository root:

```text
05_run_r100_diversity_all_datasets.sh
06_run_mquake_beam_cap512.sh
```

Requirements:

- `set -euo pipefail`;
- CPU-only;
- use `conda run -n minerva_tf2`;
- no required GPU positional argument;
- clear stage/dataset labels;
- print or preserve each generated output directory;
- primary script runs the complete seed-42 `R=100` matrix on all four datasets;
- cap script runs the MQuAKE cap-512 sensitivity.

If the deterministic beam requires a separate evaluator entry point, the launcher may call both evaluators and aggregate results afterward. Prefer correctness over forcing everything through one process.

---

## 15. Result artifact requirements

Do not overwrite historical artifacts.

New results should live in normal timestamped output directories and export machine-readable summaries.

For the post-review campaign, the canonical combined result should include enough metadata to distinguish:

```text
execution mode
sampling backend
R / candidate count
beam width
seed
action cap
Hits@1
MRR
rollout success
stochastic action-message payload
fixed-rank payload where meaningful
unique candidate statistics
answer coverage
conditional MRR
truncation
checkpoint identity
git SHAs
```

After the user runs the full scripts, append the exact new artifact paths and quantitative conclusions to:

```text
codex_ref/infocost_experiment_results_20260828.md
```

Do not rewrite or delete the historical sections in that tracker.

---

## 16. Scientific decision rule after the new experiment

Do **not** force the result to support the current manuscript narrative.

### Case A — deterministic beam matches or exceeds stochastic Top-K

If deterministic beam reaches comparable/better MRR with zero incremental stochastic-realization payload under the shared assumptions:

- candidate diversity, not stochastic communication itself, explains most/all of the observed ranking benefit within these decoders;
- flag this prominently;
- the manuscript must be reframed;
- do not hide the result or describe stochastic communication as necessary.

### Case B — deterministic beam is meaningfully worse

If Top-2/Top-4 outperform deterministic beam on ranking utility under a comparable candidate budget:

- communicated stochastic branch realization supplies utility beyond this deterministic diverse decoder family;
- this materially strengthens the paper's communication claim;
- quantify the gap and inspect whether it comes from answer coverage, conditional ranking, or both.

### Case C — different coverage/ranking tradeoffs

If beam and stochastic modes trade candidate coverage against conditional ranking:

- report the decomposition explicitly;
- do not reduce the result to a single MRR comparison;
- candidate diversity/coverage diagnostics become part of the scientific result.

---

## 17. Current definition of done

This post-review follow-up is complete when:

1. a deterministic diverse baseline is available without retraining;
2. its relationship to pinned upstream beam search is verified/documented;
3. candidate-diversity and answer-coverage diagnostics are implemented for all relevant `R>1` modes;
4. existing rate-evaluator regressions still pass;
5. no checkpoint/dataset/submodule mutation occurs;
6. CPU launchers are prepared;
7. the user has run the full seed-42 four-dataset matrix;
8. MQuAKE beam cap sensitivity is measured or explicitly shown infeasible;
9. all four datasets have a uniform NumPy unrestricted `R=100` reference;
10. the result tracker is updated after the runs;
11. manuscript interpretation is revisited only after the empirical outcome is known.

---

## 18. Historical implementation record

The original P0 and rollout-aware follow-up were completed on 2026-08-27/28. Their detailed artifacts, commands, metrics, cap-sensitivity results, and five-seed Kinship/MetaQA results are preserved in:

```text
codex_ref/infocost_experiment_results_20260828.md
```

Git history prior to this update contains the longer historical implementation brief and completed TODO record. Do not infer from those older "complete" status statements that the new post-review deterministic-diversity experiment has already been run.

The following historical guarantees remain important:

- `evaluation_infocost.py` behavior was preserved;
- `tf_policy` remained the upstream TensorFlow regression reference;
- `numpy_policy` and Top-K use seeded NumPy PCG64 sampling;
- explicit rollout/action-cap overrides are evaluation-only;
- all previous validation used `minerva_tf2`;
- no model was retrained;
- no checkpoint was modified;
- the MINERVA submodule stayed pinned at `9bf1ae998d14471c3f7c31f70969d0bbf9873329`.

---

## 19. Final instruction to Codex

Prioritize **scientific correctness and compatibility with the existing evaluator** over refactoring elegance.

Before editing, verify repository behavior directly. If the pinned upstream beam implementation conflicts with assumptions in this brief, preserve actual semantics and document the conflict rather than silently redefining the experiment.

At the end of the implementation turn, **do not commit or push**. Report:

1. root HEAD and submodule SHA;
2. whether upstream beam could be reused directly;
3. exact files changed;
4. exact new CLI mode/flags;
5. deterministic-beam communication interpretation;
6. definitions of diversity/coverage diagnostics;
7. tests and smoke-test outcomes;
8. regression status for existing modes;
9. launcher scripts created;
10. any remaining scientific/implementation caveats;
11. exact commands the user should run next.

---

## 20. Implementation status — 2026-08-29

The post-review evaluator support is **implemented and locally validated**. The full four-dataset campaign remains **pending user launch**; therefore Sections 16–17 must not yet be read as empirically resolved.

Implemented without changing the MINERVA submodule:

- opt-in `deterministic_beam` execution through an evaluation-only mirror of the pinned `TrainerNLQ.test(..., beam=True)` pruning/state-reordering semantics;
- cumulative pretrained-policy log-probability ranking, the pinned default-`argsort` tie behavior, and unchanged MINERVA `pool="max"` candidate ranking;
- `--rate_include_deterministic_beam` and `--rate_beam_width` CLI controls, both disabled/unspecified by default;
- per-question terminal diversity, candidate-answer coverage, conditional-MRR, unique-correct-terminal, and correct-rollout diagnostics for all execution modes;
- explicit `mean_question_stochastic_action_payload_bits`, equal to zero for greedy and deterministic beam under the synchronized-side-information hypothesis;
- null fixed-rank/surprisal/Shannon execution-message fields for deterministic beam rather than invented branch-message costs;
- CPU launchers `experiments/05_run_r100_diversity_all_datasets.sh` and `experiments/06_run_mquake_beam_cap512.sh`.

Local validation:

- 27/27 `unittest` tests pass in `minerva_tf2`;
- compile and shell syntax checks pass;
- one-batch Kinship `R=100`, beam-width-100 smoke completed at `saved_models/kinshiphinton/20260829_014259/rate_sweep`;
- external beam and pinned upstream beam match exactly on that batch: Hits@1 `0.96875`, MRR `0.98046875`;
- the beam smoke records mean `9.1875` unique terminal candidates/question, candidate coverage `1.0`, null coding-message fields, and zero stochastic-action payload;
- an exact pre-change Kinship `R=1` smoke comparison gives zero deltas for legacy greedy, Top-2, NumPy, and `tf_policy` utility/path/communication fields;
- checkpoint identity remained unchanged and the submodule remains pinned at `9bf1ae998d14471c3f7c31f70969d0bbf9873329`.

Pinned-upstream caveat: upstream beam always retains the requested width slots. When a state has fewer valid actions than the beam width, padded/very-low-score filler branches can be carried. The implementation preserves this behavior for regression compatibility and measures actual distinct terminal candidates; beam width 100 must not be described as 100 unique valid paths or as compute-free.

Remaining definition-of-done items are the user-run full seed-42 four-dataset matrix, MQuAKE cap-512 sensitivity, uniform NumPy `R=100` empirical rows, canonical result-tracker update with exact artifacts, and manuscript interpretation after those results exist.
