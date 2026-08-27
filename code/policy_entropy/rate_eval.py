"""Evaluation-only greedy/Top-K rollouts with MINERVA-compatible performance metrics."""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from tqdm import tqdm

from policy_entropy.rate_constraints import (
    accumulate_execution_path_log_probs,
    build_global_relation_prior,
    cross_entropy_bits,
    effective_fixed_rank_bits,
    entropy_bits_from_normalized_log_probs,
    fixed_and_gold_hop_path_costs,
    kl_bits,
    question_total_cost_bits,
    selected_surprisal_bits,
    select_actions_from_log_probs,
    shannon_integer_code_length,
    task_agnostic_local_log_probs,
    valid_action_mask,
)


QUESTION_TOTAL_SUMMARY_FIELDS = (
    "mean_question_fixed_rank_bits",
    "mean_question_surprisal_bits",
    "mean_question_shannon_code_bits",
    "mean_question_entropy_sum_bits",
    "mean_question_task_agnostic_surprisal_bits",
    "mean_question_task_agnostic_shannon_bits",
)


def resolve_evaluation_overrides(
    options: Dict[str, Any],
    test_rollouts: Optional[int] = None,
    max_num_actions: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return effective evaluation options and explicit configured/effective metadata."""
    configured = dict(options)
    effective = dict(options)

    overrides = {
        "test_rollouts": test_rollouts,
        "max_num_actions": max_num_actions,
    }
    for key, override in overrides.items():
        if override is None:
            continue
        if isinstance(override, bool) or int(override) != override or override < 1:
            raise ValueError(f"Evaluation override {key} must be a positive integer.")
        effective[key] = int(override)

    metadata = {
        "configured_test_rollouts": int(configured["test_rollouts"]),
        "effective_test_rollouts": int(effective["test_rollouts"]),
        "test_rollouts_overridden": test_rollouts is not None,
        "configured_max_num_actions": int(configured["max_num_actions"]),
        "effective_max_num_actions": int(effective["max_num_actions"]),
        "max_num_actions_overridden": max_num_actions is not None,
        "model_load_dir_unchanged": (
            effective.get("model_load_dir") == configured.get("model_load_dir")
        ),
        "checkpoint_shape_note": (
            "max_num_actions sizes graph/action arrays and non-trainable candidate axes; "
            "checkpoint compatibility is confirmed by restoring into the effective graph."
        ),
    }
    return effective, metadata


def minerva_max_pool_metrics(
    final_entities: np.ndarray,
    answer_hits: np.ndarray,
    path_log_probs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Reproduce ``TrainerNLQ.test()`` MAX-pool entity ranking exactly.

    Incorrect duplicate terminal entities do not consume additional rank positions.
    For a multi-answer question, ``answer_hits`` is already true for every gold endpoint.
    ``np.argsort`` intentionally uses NumPy's default ordering, as upstream does.
    """
    final_entities = np.asarray(final_entities)
    answer_hits = np.asarray(answer_hits, dtype=bool)
    path_log_probs = np.asarray(path_log_probs, dtype=np.float64)
    if (
        final_entities.ndim != 2
        or answer_hits.shape != final_entities.shape
        or path_log_probs.shape != final_entities.shape
    ):
        raise ValueError("final_entities, answer_hits, and path_log_probs must share [B, R].")

    sorted_indices = np.argsort(-path_log_probs)
    answer_ranks = np.full(final_entities.shape[0], -1, dtype=np.int64)
    reciprocal_ranks = np.zeros(final_entities.shape[0], dtype=np.float64)

    for question_idx in range(final_entities.shape[0]):
        seen = set()
        unique_incorrect_rank = 0
        for rollout_idx in sorted_indices[question_idx]:
            if answer_hits[question_idx, rollout_idx]:
                answer_ranks[question_idx] = unique_incorrect_rank
                reciprocal_ranks[question_idx] = 1.0 / (unique_incorrect_rank + 1)
                break
            terminal_entity = final_entities[question_idx, rollout_idx]
            if terminal_entity not in seen:
                seen.add(terminal_entity)
                unique_incorrect_rank += 1

    return {
        "sorted_indices": sorted_indices,
        "answer_ranks": answer_ranks,
        "reciprocal_ranks": reciprocal_ranks,
        "hits_at_1": answer_ranks == 0,
    }


def recover_raw_action_counts(grapher: Any) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Recover pre-cap action counts and validate them against the grapher's array store.

    The reconstruction follows ``RelationEntityGrapher.create_graph``: a NO_OP slot
    (and configured STOP/RESTART slots) is added for each head in the triple file,
    then inverse relations are excluded in directed mode before applying the cap.
    If validation fails, counts are withheld instead of being guessed.
    """
    metadata: Dict[str, Any] = {
        "available": False,
        "source": getattr(grapher, "triple_store", None),
        "limitation": None,
    }
    try:
        num_entities = grapher.array_store.shape[0]
        max_num_actions = grapher.array_store.shape[1]
        graph_relation_counts = np.zeros(num_entities, dtype=np.int64)
        head_seen = np.zeros(num_entities, dtype=bool)

        with open(grapher.triple_store, "r", encoding="utf-8") as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter="\t")
            for line in triple_file:
                if len(line) != 3:
                    raise ValueError(f"Expected three columns in graph row, received {len(line)}.")
                head_entity = grapher.entity_vocab[line[0]]
                relation = grapher.relation_vocab[line[1]]
                head_seen[head_entity] = True
                if grapher.use_directed_graph and relation in grapher.inverse_tokens:
                    continue
                graph_relation_counts[head_entity] += 1

        special_slots = 1 + int(grapher.use_stop_signal) + int(grapher.use_restart_signal)
        raw_counts = graph_relation_counts + head_seen.astype(np.int64) * special_slots

        retained_counts = np.sum(
            grapher.array_store[:, :, 1] != grapher.rPAD, axis=1
        ).astype(np.int64)
        expected_retained = np.minimum(raw_counts, max_num_actions)
        mismatch = np.flatnonzero(retained_counts != expected_retained)
        if mismatch.size:
            metadata["limitation"] = (
                "Raw-degree reconstruction did not match the active array_store for "
                f"{mismatch.size} entities; truncation statistics are omitted."
            )
            metadata["mismatch_entity_count"] = int(mismatch.size)
            return None, metadata

        metadata.update(
            {
                "available": True,
                "max_raw_valid_action_count": int(raw_counts.max(initial=0)),
                "special_slots_per_head": int(special_slots),
                "directed_inverse_filtering": bool(grapher.use_directed_graph),
                "max_num_actions": int(max_num_actions),
            }
        )
        return raw_counts, metadata
    except (OSError, KeyError, ValueError, AttributeError) as exc:
        metadata["limitation"] = f"Raw-degree reconstruction unavailable: {exc}"
        return None, metadata


def build_graph_structural_relation_prior(
    trainer: Any,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the capped full-/train-graph structural relation prior without QA labels."""
    invalid_relation_ids = [
        trainer.relation_vocab["PAD"],
        trainer.relation_vocab["UNK"],
        trainer.relation_vocab["DUMMY_START_RELATION"],
    ]
    prior = build_global_relation_prior(
        trainer.environment.grapher.array_store,
        num_relations=len(trainer.relation_vocab),
        invalid_relation_ids=invalid_relation_ids,
        alpha=alpha,
    )
    return prior, {
        "kind": "graph-structural",
        "representation": "capped array_store",
        "uses_question_embedding": False,
        "uses_answer_labels": False,
        "uses_evaluation_action_choices": False,
        "alpha": float(alpha),
        "graph_path": trainer.environment.grapher.triple_store,
    }


def _reshape_step(values: Iterable[np.ndarray], batch_size: int, rollouts: int) -> np.ndarray:
    stacked = np.stack(list(values), axis=1)
    return stacked.reshape(batch_size, rollouts, stacked.shape[1])


def _path_metric_arrays(
    trainer: Any,
    episode: Any,
    entity_trajectory: np.ndarray,
    relation_trajectory: np.ndarray,
    sorted_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute upstream path/relation edit distances from the top-scoring rollout."""
    batch_size, _, horizon_plus_one = entity_trajectory.shape
    path_edit_distance = np.full(batch_size, np.nan, dtype=np.float64)
    relation_edit_distance = np.full(batch_size, np.nan, dtype=np.float64)

    for question_idx in range(batch_size):
        rollout_idx = sorted_indices[question_idx, 0]
        merged_path_raw = [
            [
                entity_trajectory[question_idx, rollout_idx, step],
                relation_trajectory[question_idx, rollout_idx, step],
                entity_trajectory[question_idx, rollout_idx, step + 1],
            ]
            for step in range(horizon_plus_one - 1)
        ]
        merged_path = episode.clean_pred_path_for_eval(
            merged_path_raw, policy=trainer.path_segment_policy
        )
        relations_path = [step[1] for step in merged_path]

        if trainer.environment.has_paths():
            path_edit_distance[question_idx] = episode.get_path_edit_distance(
                merged_path, question_idx
            )
        if trainer.environment.has_paths_or_keys():
            relation_edit_distance[question_idx] = episode.get_relation_edit_distance(
                relations_path, question_idx
            )

    return path_edit_distance, relation_edit_distance


def collect_rate_single_episode(
    trainer: Any,
    sess: Any,
    episode: Any,
    action_mode: str = "tf_policy",
    top_k: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    relation_prior: Optional[np.ndarray] = None,
    raw_action_counts: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run one fixed-horizon episode with an optional NumPy action override."""
    if action_mode not in {"tf_policy", "greedy", "topk", "numpy_policy"}:
        raise ValueError(
            "action_mode must be 'tf_policy', 'greedy', 'topk', or 'numpy_policy'."
        )
    if action_mode == "topk" and top_k is None:
        raise ValueError("top_k is required for action_mode='topk'.")

    batch_size = episode.no_examples
    rollouts = trainer.test_rollouts
    horizon = trainer.path_length
    flat_size = batch_size * rollouts

    state = episode.get_state()
    mem_shape = trainer.agent.get_mem_shape()
    agent_mem = np.zeros(
        (mem_shape[0], mem_shape[1], flat_size, mem_shape[3]), dtype=np.float32
    )
    previous_relation = np.full(
        flat_size, trainer.relation_vocab["DUMMY_START_RELATION"], dtype=np.int64
    )
    constant_feed = {
        trainer.range_arr: np.arange(flat_size, dtype=np.int32),
        trainer.question_embedding: episode.get_question_embedding(),
    }

    per_step = defaultdict(list)
    entity_history = []
    relation_history = []
    path_log_prob = np.zeros(flat_size, dtype=np.float64)

    for _ in range(horizon):
        entity_history.append(np.asarray(state["current_entities"]).copy())
        feed_dict = {
            **constant_feed,
            trainer.next_relations: state["next_relations"],
            trainer.next_entities: state["next_entities"],
            trainer.current_entities: state["current_entities"],
            trainer.prev_state: agent_mem,
            trainer.prev_relation: previous_relation,
        }

        if action_mode == "tf_policy":
            _, agent_mem, original_log_probs, action_idx, chosen_relation = sess.run(
                [
                    trainer.test_loss,
                    trainer.test_state,
                    trainer.test_logits,
                    trainer.test_action_idx,
                    trainer.chosen_relation,
                ],
                feed_dict=feed_dict,
            )
        else:
            agent_mem, original_log_probs = sess.run(
                [trainer.test_state, trainer.test_logits], feed_dict=feed_dict
            )

        original_log_probs = np.asarray(original_log_probs, dtype=np.float64)
        current_valid_mask = valid_action_mask(
            state["next_relations"], trainer.relation_vocab["PAD"]
        )

        if action_mode == "tf_policy":
            action_idx = np.asarray(action_idx, dtype=np.int32)
            if not np.all(current_valid_mask[np.arange(flat_size), action_idx]):
                raise RuntimeError("TensorFlow selected an action outside its relation-PAD mask.")
            execution_log_probs = original_log_probs
            retained_mask = current_valid_mask
            chosen_relation = np.asarray(chosen_relation, dtype=np.int64)
        else:
            action_idx, execution_log_probs, retained_mask = select_actions_from_log_probs(
                original_log_probs,
                current_valid_mask,
                mode=action_mode,
                top_k=top_k,
                rng=rng,
            )
            chosen_relation = state["next_relations"][np.arange(flat_size), action_idx]

        execution_surprisal = selected_surprisal_bits(execution_log_probs, action_idx)
        original_surprisal = selected_surprisal_bits(original_log_probs, action_idx)
        per_step["entropy_bits"].append(
            entropy_bits_from_normalized_log_probs(execution_log_probs)
        )
        per_step["original_policy_entropy_bits"].append(
            entropy_bits_from_normalized_log_probs(original_log_probs)
        )
        per_step["surprisal_bits"].append(execution_surprisal)
        per_step["original_policy_selected_surprisal_bits"].append(original_surprisal)
        per_step["shannon_code_bits"].append(
            shannon_integer_code_length(execution_surprisal)
        )
        per_step["original_policy_selected_shannon_code_bits"].append(
            shannon_integer_code_length(original_surprisal)
        )
        per_step["valid_action_counts"].append(current_valid_mask.sum(axis=1))
        per_step["effective_support"].append(retained_mask.sum(axis=1))
        per_step["fixed_budget_bits"].append(
            effective_fixed_rank_bits(retained_mask.sum(axis=1))
        )

        if relation_prior is not None:
            agnostic_log_probs = task_agnostic_local_log_probs(
                state["next_relations"], current_valid_mask, relation_prior
            )
            agnostic_surprisal = selected_surprisal_bits(agnostic_log_probs, action_idx)
            per_step["task_agnostic_surprisal_bits"].append(agnostic_surprisal)
            per_step["task_agnostic_shannon_bits"].append(
                shannon_integer_code_length(agnostic_surprisal)
            )
            per_step["execution_policy_vs_task_agnostic_cross_entropy_bits"].append(
                cross_entropy_bits(execution_log_probs, agnostic_log_probs)
            )
            per_step["execution_policy_vs_task_agnostic_kl_bits"].append(
                kl_bits(execution_log_probs, agnostic_log_probs)
            )
            per_step["original_policy_vs_task_agnostic_cross_entropy_bits"].append(
                cross_entropy_bits(original_log_probs, agnostic_log_probs)
            )
            per_step["original_policy_vs_task_agnostic_kl_bits"].append(
                kl_bits(original_log_probs, agnostic_log_probs)
            )

        if raw_action_counts is not None:
            raw_counts = raw_action_counts[np.asarray(state["current_entities"], dtype=np.int64)]
            per_step["raw_action_counts"].append(raw_counts)
            per_step["action_cap_truncated"].append(
                raw_counts > trainer.max_num_actions
            )

        path_log_prob = accumulate_execution_path_log_probs(
            path_log_prob, execution_log_probs, action_idx
        )
        relation_history.append(np.asarray(chosen_relation).copy())
        previous_relation = chosen_relation
        state = episode(action_idx)

    entity_history.append(np.asarray(state["current_entities"]).copy())
    _, answer_hits_flat = episode.get_reward()
    answer_hits = np.asarray(answer_hits_flat, dtype=bool).reshape(batch_size, rollouts)
    final_entities = np.asarray(state["current_entities"]).reshape(batch_size, rollouts)
    path_log_prob = path_log_prob.reshape(batch_size, rollouts)

    result: Dict[str, Any] = {
        key: _reshape_step(values, batch_size, rollouts)
        for key, values in per_step.items()
    }
    result.update(
        {
            "final_entities": final_entities,
            "answer_hits": answer_hits,
            "path_log_prob": path_log_prob,
            "gold_hops": np.asarray(
                [episode.get_path_length(idx) for idx in range(batch_size)], dtype=np.int64
            ),
            "entity_trajectory": np.stack(entity_history, axis=1).reshape(
                batch_size, rollouts, horizon + 1
            ),
            "relation_trajectory": np.stack(relation_history, axis=1).reshape(
                batch_size, rollouts, horizon
            ),
        }
    )

    ranking = minerva_max_pool_metrics(final_entities, answer_hits, path_log_prob)
    result.update(ranking)
    path_ed, relation_ed = _path_metric_arrays(
        trainer,
        episode,
        result["entity_trajectory"],
        result["relation_trajectory"],
        ranking["sorted_indices"],
    )
    result["path_edit_distance"] = path_ed
    result["relation_edit_distance"] = relation_ed

    path_cost_sources = {
        "entropy": "entropy_bits",
        "fixed_rank": "fixed_budget_bits",
        "original_policy_entropy": "original_policy_entropy_bits",
        "surprisal": "surprisal_bits",
        "shannon_code": "shannon_code_bits",
        "task_agnostic_surprisal": "task_agnostic_surprisal_bits",
        "task_agnostic_shannon": "task_agnostic_shannon_bits",
    }
    for prefix, step_key in path_cost_sources.items():
        if step_key in result:
            fixed, gold = fixed_and_gold_hop_path_costs(result[step_key], result["gold_hops"])
            result[f"path_{prefix}_fixed_horizon_bits"] = fixed
            result[f"path_{prefix}_gold_hops_bits"] = gold
    question_cost_sources = {
        "question_fixed_rank_bits": "fixed_budget_bits",
        "question_surprisal_bits": "surprisal_bits",
        "question_shannon_code_bits": "shannon_code_bits",
        "question_entropy_sum_bits": "entropy_bits",
        "question_task_agnostic_surprisal_bits": "task_agnostic_surprisal_bits",
        "question_task_agnostic_shannon_bits": "task_agnostic_shannon_bits",
    }
    for output_key, step_key in question_cost_sources.items():
        if step_key in result:
            result[output_key] = question_total_cost_bits(result[step_key])

    return result


class _MeanAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    def add(
        self,
        name: str,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        array = np.asarray(values, dtype=np.float64)
        selected = np.ones(array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if selected.shape != array.shape:
            raise ValueError(f"Mask shape for {name} does not match values.")
        selected &= np.isfinite(array)
        self.sums[name] += float(array[selected].sum())
        self.counts[name] += int(selected.sum())

    def mean(self, name: str) -> Optional[float]:
        if self.counts[name] == 0:
            return None
        return self.sums[name] / self.counts[name]


def evaluate_rate_mode(
    trainer: Any,
    sess: Any,
    action_mode: str,
    seed: int,
    top_k: Optional[int] = None,
    relation_prior: Optional[np.ndarray] = None,
    raw_action_counts: Optional[np.ndarray] = None,
    raw_action_metadata: Optional[Dict[str, Any]] = None,
    mode: str = "test",
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Stream a complete evaluation split and return one common summary row."""
    trainer.environment.change_mode(mode)
    trainer.environment.change_test_rollouts(trainer.test_rollouts)
    rng = np.random.default_rng(seed)
    accumulator = _MeanAccumulator()
    question_count = 0
    per_hop_truncated = np.zeros(trainer.path_length, dtype=np.float64)
    per_hop_visited = np.zeros(trainer.path_length, dtype=np.int64)

    question_num = trainer.environment.batcher.get_question_num()
    test_batch_size = trainer.environment.batcher.test_batch_size
    total_batches = (question_num + test_batch_size - 1) // test_batch_size
    num_batches = total_batches if max_batches is None else min(max_batches, total_batches)

    episodes = trainer.environment.get_episodes()
    for batch_idx, episode in enumerate(
        tqdm(episodes, desc=f"Evaluating rate mode={action_mode} K={top_k}", total=num_batches)
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = collect_rate_single_episode(
            trainer=trainer,
            sess=sess,
            episode=episode,
            action_mode=action_mode,
            top_k=top_k,
            rng=rng,
            relation_prior=relation_prior,
            raw_action_counts=raw_action_counts,
        )
        question_count += int(batch["final_entities"].shape[0])

        direct_metrics = {
            "mean_valid_actions": "valid_action_counts",
            "mean_effective_support": "effective_support",
            "mean_fixed_budget_bits": "fixed_budget_bits",
            "mean_step_entropy_bits": "entropy_bits",
            "mean_step_original_policy_entropy_bits": "original_policy_entropy_bits",
            "mean_step_surprisal_bits": "surprisal_bits",
            "mean_step_original_policy_selected_surprisal_bits": "original_policy_selected_surprisal_bits",
            "mean_step_shannon_code_bits": "shannon_code_bits",
            "mean_step_original_policy_selected_shannon_code_bits": "original_policy_selected_shannon_code_bits",
            "mean_task_agnostic_surprisal_bits": "task_agnostic_surprisal_bits",
            "mean_task_agnostic_shannon_bits": "task_agnostic_shannon_bits",
            "mean_execution_policy_vs_task_agnostic_cross_entropy_bits": "execution_policy_vs_task_agnostic_cross_entropy_bits",
            "mean_execution_policy_vs_task_agnostic_kl_bits": "execution_policy_vs_task_agnostic_kl_bits",
            "mean_original_policy_vs_task_agnostic_cross_entropy_bits": "original_policy_vs_task_agnostic_cross_entropy_bits",
            "mean_original_policy_vs_task_agnostic_kl_bits": "original_policy_vs_task_agnostic_kl_bits",
            "mean_path_entropy_fixed_horizon_bits": "path_entropy_fixed_horizon_bits",
            "mean_path_entropy_gold_hops_bits": "path_entropy_gold_hops_bits",
            "mean_path_original_policy_entropy_fixed_horizon_bits": "path_original_policy_entropy_fixed_horizon_bits",
            "mean_path_original_policy_entropy_gold_hops_bits": "path_original_policy_entropy_gold_hops_bits",
            "mean_path_surprisal_fixed_horizon_bits": "path_surprisal_fixed_horizon_bits",
            "mean_path_surprisal_gold_hops_bits": "path_surprisal_gold_hops_bits",
            "mean_path_shannon_code_fixed_horizon_bits": "path_shannon_code_fixed_horizon_bits",
            "mean_path_shannon_code_gold_hops_bits": "path_shannon_code_gold_hops_bits",
            "mean_path_fixed_rank_bits": "path_fixed_rank_fixed_horizon_bits",
            "mean_question_fixed_rank_bits": "question_fixed_rank_bits",
            "mean_question_surprisal_bits": "question_surprisal_bits",
            "mean_question_shannon_code_bits": "question_shannon_code_bits",
            "mean_question_entropy_sum_bits": "question_entropy_sum_bits",
            "mean_question_task_agnostic_surprisal_bits": "question_task_agnostic_surprisal_bits",
            "mean_question_task_agnostic_shannon_bits": "question_task_agnostic_shannon_bits",
            "mean_path_task_agnostic_surprisal_fixed_horizon_bits": "path_task_agnostic_surprisal_fixed_horizon_bits",
            "mean_path_task_agnostic_surprisal_gold_hops_bits": "path_task_agnostic_surprisal_gold_hops_bits",
            "mean_path_task_agnostic_shannon_fixed_horizon_bits": "path_task_agnostic_shannon_fixed_horizon_bits",
            "mean_path_task_agnostic_shannon_gold_hops_bits": "path_task_agnostic_shannon_gold_hops_bits",
            "ped": "path_edit_distance",
            "relation_edit_distance": "relation_edit_distance",
        }
        for output_name, batch_key in direct_metrics.items():
            if batch_key in batch:
                accumulator.add(output_name, batch[batch_key])

        accumulator.add("hits_at_1", batch["hits_at_1"])
        accumulator.add("mrr", batch["reciprocal_ranks"])
        accumulator.add("rollout_success_rate", batch["answer_hits"])

        success_mask = batch["answer_hits"]
        failure_mask = ~success_mask
        conditional_sources = {
            "path_entropy": "path_entropy_fixed_horizon_bits",
            "path_surprisal": "path_surprisal_fixed_horizon_bits",
            "path_shannon_code": "path_shannon_code_fixed_horizon_bits",
        }
        for output_stem, batch_key in conditional_sources.items():
            accumulator.add(f"success_mean_{output_stem}_bits", batch[batch_key], success_mask)
            accumulator.add(f"failure_mean_{output_stem}_bits", batch[batch_key], failure_mask)

        if "action_cap_truncated" in batch:
            accumulator.add("truncated_state_fraction", batch["action_cap_truncated"])
            truncated = batch["action_cap_truncated"].astype(bool)
            accumulator.add(
                "mean_raw_count_for_truncated_states",
                batch["raw_action_counts"],
                truncated,
            )
            accumulator.add(
                "mean_retained_count_for_truncated_states",
                batch["valid_action_counts"],
                truncated,
            )
            per_hop_truncated += truncated.sum(axis=(0, 1))
            per_hop_visited += np.prod(truncated.shape[:2])

    nominal_budget = None
    if action_mode == "greedy":
        nominal_budget = 0.0
    elif action_mode == "topk":
        nominal_budget = float(np.log2(top_k))

    summary: Dict[str, Any] = {
        "mode": action_mode,
        "top_k": int(top_k) if top_k is not None else None,
        "nominal_budget_bits": nominal_budget,
        "num_questions": int(question_count),
        "num_rollouts": int(trainer.test_rollouts),
        "path_length": int(trainer.path_length),
        "seed": int(seed),
        "sampling_backend": (
            "deterministic argmax"
            if action_mode == "greedy"
            else "numpy.random.default_rng (PCG64)"
            if action_mode in {"topk", "numpy_policy"}
            else "upstream TensorFlow categorical"
        ),
        "action_payload_interpretation": (
            "0 bits/hop with synchronized side information"
            if action_mode == "greedy"
            else "fixed local rank within retained support"
            if action_mode == "topk"
            else "unrestricted pretrained NumPy policy"
            if action_mode == "numpy_policy"
            else "unrestricted pretrained TensorFlow policy"
        ),
    }

    output_metrics = [
        "hits_at_1",
        "mrr",
        "ped",
        "relation_edit_distance",
        "rollout_success_rate",
        "mean_valid_actions",
        "mean_effective_support",
        "mean_fixed_budget_bits",
        "mean_step_entropy_bits",
        "mean_step_original_policy_entropy_bits",
        "mean_step_surprisal_bits",
        "mean_step_original_policy_selected_surprisal_bits",
        "mean_step_shannon_code_bits",
        "mean_step_original_policy_selected_shannon_code_bits",
        "mean_path_entropy_fixed_horizon_bits",
        "mean_path_entropy_gold_hops_bits",
        "mean_path_original_policy_entropy_fixed_horizon_bits",
        "mean_path_original_policy_entropy_gold_hops_bits",
        "mean_path_surprisal_fixed_horizon_bits",
        "mean_path_surprisal_gold_hops_bits",
        "mean_path_shannon_code_fixed_horizon_bits",
        "mean_path_shannon_code_gold_hops_bits",
        "mean_path_fixed_rank_bits",
        *QUESTION_TOTAL_SUMMARY_FIELDS,
        "mean_task_agnostic_surprisal_bits",
        "mean_task_agnostic_shannon_bits",
        "mean_path_task_agnostic_surprisal_fixed_horizon_bits",
        "mean_path_task_agnostic_surprisal_gold_hops_bits",
        "mean_path_task_agnostic_shannon_fixed_horizon_bits",
        "mean_path_task_agnostic_shannon_gold_hops_bits",
        "mean_execution_policy_vs_task_agnostic_cross_entropy_bits",
        "mean_execution_policy_vs_task_agnostic_kl_bits",
        "mean_original_policy_vs_task_agnostic_cross_entropy_bits",
        "mean_original_policy_vs_task_agnostic_kl_bits",
        "success_mean_path_entropy_bits",
        "failure_mean_path_entropy_bits",
        "success_mean_path_surprisal_bits",
        "failure_mean_path_surprisal_bits",
        "success_mean_path_shannon_code_bits",
        "failure_mean_path_shannon_code_bits",
        "truncated_state_fraction",
        "mean_raw_count_for_truncated_states",
        "mean_retained_count_for_truncated_states",
    ]
    summary.update({name: accumulator.mean(name) for name in output_metrics})

    is_single_trajectory = trainer.test_rollouts == 1
    summary["single_rollout_success_rate"] = (
        summary["rollout_success_rate"] if is_single_trajectory else None
    )
    summary["utility_protocol"] = (
        "single_trajectory" if is_single_trajectory else "rollout_ensemble_candidate_ranking"
    )
    summary["communication_scope"] = (
        "mean_question_* sums all rollout and hop costs; mean_path_* averages one path; "
        "mean_step_* averages one rollout-hop position."
    )
    if is_single_trajectory and not np.isclose(
        summary["hits_at_1"], summary["single_rollout_success_rate"], atol=1e-12
    ):
        raise RuntimeError("R=1 protocol invariant failed: Hits@1 != rollout success.")
    if action_mode == "greedy" and not np.isclose(
        summary["mrr"], summary["hits_at_1"], atol=1e-12
    ):
        raise RuntimeError("Greedy protocol invariant failed: MRR != Hits@1.")

    # Compatibility aliases requested by the brief's minimum common schema.
    summary["mean_path_surprisal_bits"] = summary["mean_path_surprisal_fixed_horizon_bits"]
    summary["mean_path_shannon_code_bits"] = summary[
        "mean_path_shannon_code_fixed_horizon_bits"
    ]
    summary["mean_policy_vs_task_agnostic_kl_bits"] = summary[
        "mean_execution_policy_vs_task_agnostic_kl_bits"
    ]
    summary["success_mean_path_code_bits"] = summary[
        "success_mean_path_shannon_code_bits"
    ]
    summary["failure_mean_path_code_bits"] = summary[
        "failure_mean_path_shannon_code_bits"
    ]
    summary["success_rollout_count"] = accumulator.counts[
        "success_mean_path_shannon_code_bits"
    ]
    summary["failure_rollout_count"] = accumulator.counts[
        "failure_mean_path_shannon_code_bits"
    ]

    if np.all(per_hop_visited > 0):
        summary["per_hop_truncation_rate"] = (
            per_hop_truncated / per_hop_visited
        ).tolist()
    else:
        summary["per_hop_truncation_rate"] = None
    summary["max_raw_valid_action_count"] = (
        raw_action_metadata.get("max_raw_valid_action_count")
        if raw_action_metadata and raw_action_metadata.get("available")
        else None
    )
    summary["truncation_diagnostic_limitation"] = (
        raw_action_metadata.get("limitation") if raw_action_metadata else "Not requested."
    )
    return summary
