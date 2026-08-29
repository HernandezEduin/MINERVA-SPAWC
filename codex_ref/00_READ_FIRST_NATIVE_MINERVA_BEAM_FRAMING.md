# Authoritative framing clarification — native MINERVA beam and `R=100`

**Date:** 2026-08-29  
**Applies to:** MINERVA-InfoCost reference material and future manuscript-support analysis  
**Status:** This note supersedes any conflicting *framing* in older `codex_ref` files. It does **not** replace or alter recorded experimental values, artifacts, or implementation results.

## 1. Provenance of `R=100`

The `test_rollouts = 100` setting predates the InfoCost study. It is the standard MINERVA/Theseus test-time evaluation configuration used so that MINERVA's rollout-based ranked-candidate metrics, especially MRR (and Hits@K), are meaningful. With only one candidate, MRR collapses to the same binary outcome as Hits@1; with very few candidates it gives an impoverished view of ranking behavior.

Therefore, do **not** describe `R=100` as a new InfoCost communication objective introduced specifically to create a separate multi-rollout task.

## 2. Native MINERVA / Theseus baseline

The MINERVA results reported in Theseus use:

```text
test_rollouts = 100
beam = True
```

Thus the deterministic width-100 beam evaluated in the 2026-08-29 InfoCost follow-up is the **native MINERVA beam-search inference baseline**, not a newly invented deterministic-diversity decoder.

Pinned upstream `TrainerNLQ.test(..., beam=True)` keeps the best partial paths under cumulative policy log-probability and prunes/reorders beam states at every hop. Strong parents may occupy several surviving slots through different outgoing expansions; weaker partial paths are removed. The final 100 beam slots are not 100 independent rollouts and are not guaranteed to be 100 unique complete paths.

Preferred terminology after definition:

```text
MINERVA beam search (beam width 100)
```

or simply:

```text
Beam
```

Do **not** imply that Beam-100 is a new InfoCost algorithm.

## 3. What InfoCost changes relative to native MINERVA evaluation

The original InfoCost `R=100` rate sweeps intentionally use:

```text
beam = False
```

so that the 100 trajectory slots are generated through stochastic policy execution and the realized action indices can be communication-accounted.

The correct conceptual comparison is:

```text
Native MINERVA / Theseus evaluation:
    R = 100
    beam = True
    deterministic beam-search candidate generation

InfoCost stochastic evaluation:
    R = 100
    beam = False
    stochastic trajectory generation
    realized actions are rate-accounted

InfoCost single-trajectory diagnostic:
    R = 1
    beam = False
    isolates one executed stochastic trajectory and its action-message cost
```

`R=1` is the InfoCost-specific diagnostic that isolates one executed trajectory. It should not be presented as the historical MINERVA test protocol.

## 4. Reinterpretation of the 2026-08-29 follow-up

Older planning text frames the follow-up as testing whether a hypothetical deterministic diverse decoder could recover stochastic ranking gains without stochastic realization messages.

That framing is superseded.

The experiment should instead be understood as **restoring the standard native MINERVA/Theseus beam baseline to the InfoCost comparison** and asking:

> How do the communication-accounted stochastic execution modes compare with MINERVA's standard deterministic beam-search inference under the same fixed pretrained policy and 100-slot test-time candidate budget?

The external evaluation-only beam implementation in `code/policy_entropy/rate_eval.py` mirrors pinned upstream beam semantics because the upstream aggregate test API does not expose all terminal-candidate diagnostics needed by InfoCost. Its scientific role is to reproduce the native baseline while exposing additional diagnostics; it is not a new decoding method.

The regression check against pinned upstream beam remains important evidence that this mirror is faithful.

## 5. Primary utility metrics remain MINERVA Hits@1 and MRR

Keep the established MINERVA utility metrics as the primary scientific metrics:

```text
Hits@1
MRR
```

The 2026-08-29 follow-up also logs InfoCost-specific diagnostics such as:

```text
candidate_answer_coverage
mean_unique_terminal_candidates
mrr_given_answer_coverage
```

These are useful analysis diagnostics, not automatically new manuscript metrics.

In particular, `candidate_answer_coverage` means only:

```text
at least one of the final candidate slots ends at a valid answer
```

It should not be confused with MINERVA's existing multi-answer coverage / answer precision-recall-F1 machinery.

Do not promote these diagnostics to headline metrics solely because they were added after the beam comparison.

## 6. Communication interpretation of native beam

Under the conditional shared-side-information accounting, native deterministic beam samples no stochastic action realization. Therefore:

```text
incremental stochastic action-realization payload = 0
```

This does **not** imply zero total communication or zero system cost. Beam still requires deterministic search, beam-state maintenance, branch-local policy evaluation, candidate processing, and whatever synchronization is assumed for branch states/local action interfaces.

Do not fabricate fixed-rank, surprisal, Shannon, or entropy-coded branch-message values for deterministic beam. Keep conceptually undefined stochastic coding fields null/NA where appropriate.

## 7. Relationship to existing canonical tracker and implementation plan

The following files remain authoritative for their numerical/implementation contents:

```text
codex_ref/infocost_experiment_results_20260828.md
codex_ref/codex_minerva_infocost_no_retraining_plan.md
```

However, any statements in those files that frame:

- `R=100` as a new InfoCost multi-rollout objective;
- the 2026-08-29 beam comparison as a new deterministic-diversity algorithm;
- native beam as merely a hypothetical reviewer-requested alternative;
- coverage/diversity diagnostics as required headline paper metrics;

are superseded by this clarification.

Historical values, artifact paths, seed summaries, backend notes, cap-sensitivity results, and communication measurements remain unchanged.

## 8. Required framing for future Codex analysis

Before future manuscript-support or scientific-interpretation work, use this provenance:

1. `R=100` comes from standard MINERVA/Theseus evaluation and is retained so rollout-based ranking metrics such as MRR are informative.
2. Standard Theseus MINERVA inference uses native beam search with 100 test rollouts/beam slots.
3. InfoCost disables beam for its stochastic `R=100` communication experiments.
4. InfoCost adds `R=1` to isolate single-trajectory utility and communication.
5. The 2026-08-29 beam results restore the native MINERVA baseline to the comparison; they do not introduce a new beam algorithm.
6. Hits@1 and MRR remain the primary utility metrics; candidate diversity/coverage quantities are optional diagnostics.
7. Do not change experimental numbers solely because the framing has been corrected.
