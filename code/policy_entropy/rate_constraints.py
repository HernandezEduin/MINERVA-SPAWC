"""Pure NumPy helpers for evaluation-only action-rate constraints and coding costs."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


LOG2 = np.log(2.0)


def valid_action_mask(
    next_relations: np.ndarray,
    pad_relation_id: int,
) -> np.ndarray:
    """Return MINERVA's executable-action mask (the policy masks relation PAD only)."""
    next_relations = np.asarray(next_relations)
    if next_relations.ndim != 2:
        raise ValueError("next_relations must have shape [num_rollouts, num_actions].")
    mask = next_relations != pad_relation_id
    if np.any(mask.sum(axis=1) == 0):
        rows = np.flatnonzero(mask.sum(axis=1) == 0)
        raise ValueError(f"Found all-PAD action rows at indices {rows[:10].tolist()}.")
    return mask


def _validate_log_prob_inputs(
    log_probs: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if log_probs.ndim != 2 or mask.shape != log_probs.shape:
        raise ValueError("log_probs and valid_mask must have the same two-dimensional shape.")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("Every row must contain at least one valid action.")
    if np.any(~np.isfinite(log_probs[mask])):
        raise ValueError("Every valid action must have a finite log probability.")
    return log_probs, mask


def _renormalize_masked_log_probs(
    log_probs: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    log_probs, mask = _validate_log_prob_inputs(log_probs, mask)
    result = np.full(log_probs.shape, -np.inf, dtype=np.float64)
    masked = np.where(mask, log_probs, -np.inf)
    row_max = np.max(masked, axis=1, keepdims=True)
    log_normalizer = row_max + np.log(
        np.sum(np.where(mask, np.exp(log_probs - row_max), 0.0), axis=1, keepdims=True)
    )
    result[mask] = (log_probs - log_normalizer)[mask]
    return result


def normalized_valid_log_probs(
    log_probs: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Normalize a policy over exactly the valid MINERVA action support."""
    return _renormalize_masked_log_probs(log_probs, valid_mask)


def select_greedy(
    log_probs: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Select the maximum-probability valid action; ties use the lowest local index."""
    log_probs, valid_mask = _validate_log_prob_inputs(log_probs, valid_mask)
    return np.argmax(np.where(valid_mask, log_probs, -np.inf), axis=1).astype(np.int32)


def topk_renormalized_log_probs(
    log_probs: np.ndarray,
    valid_mask: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the row-wise top ``min(top_k, valid_count)`` actions and renormalize."""
    if top_k is None or int(top_k) != top_k or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    log_probs, valid_mask = _validate_log_prob_inputs(log_probs, valid_mask)

    retained_mask = np.zeros_like(valid_mask, dtype=bool)
    masked_scores = np.where(valid_mask, log_probs, -np.inf)
    # Stable descending order makes equal-score ties prefer the lowest local index.
    ordered = np.argsort(-masked_scores, axis=1, kind="stable")
    valid_counts = valid_mask.sum(axis=1)
    for row, valid_count in enumerate(valid_counts):
        retained_mask[row, ordered[row, : min(int(top_k), int(valid_count))]] = True

    return _renormalize_masked_log_probs(log_probs, retained_mask), retained_mask


def sample_log_probs(
    normalized_log_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one action per row from normalized log probabilities using ``rng``."""
    if rng is None:
        raise ValueError("A seeded NumPy Generator is required for sampling.")
    normalized_log_probs = np.asarray(normalized_log_probs, dtype=np.float64)
    if normalized_log_probs.ndim != 2:
        raise ValueError("normalized_log_probs must have shape [num_rollouts, num_actions].")

    probabilities = np.exp(normalized_log_probs)
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Each row of normalized_log_probs must sum to one.")

    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(probabilities.shape[0])
    action_idx = np.sum(draws[:, None] > cumulative, axis=1)
    action_idx = np.minimum(action_idx, probabilities.shape[1] - 1).astype(np.int32)
    if np.any(probabilities[np.arange(probabilities.shape[0]), action_idx] <= 0.0):
        raise RuntimeError("Sampling selected an action outside the positive-probability support.")
    return action_idx


def deterministic_log_probs(action_idx: np.ndarray, num_actions: int) -> np.ndarray:
    """Construct a deterministic execution policy concentrated on ``action_idx``."""
    action_idx = np.asarray(action_idx, dtype=np.int64)
    if action_idx.ndim != 1 or np.any(action_idx < 0) or np.any(action_idx >= num_actions):
        raise ValueError("action_idx must be a valid one-dimensional action-index array.")
    result = np.full((action_idx.shape[0], num_actions), -np.inf, dtype=np.float64)
    result[np.arange(action_idx.shape[0]), action_idx] = 0.0
    return result


def select_actions_from_log_probs(
    log_probs: np.ndarray,
    valid_mask: np.ndarray,
    mode: str,
    top_k: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected indices, execution log probabilities, and retained support."""
    if mode == "greedy":
        action_idx = select_greedy(log_probs, valid_mask)
        execution_log_probs = deterministic_log_probs(action_idx, log_probs.shape[1])
        retained_mask = np.isfinite(execution_log_probs)
    elif mode == "topk":
        execution_log_probs, retained_mask = topk_renormalized_log_probs(
            log_probs, valid_mask, top_k
        )
        action_idx = sample_log_probs(execution_log_probs, rng)
    elif mode == "numpy_policy":
        execution_log_probs = normalized_valid_log_probs(log_probs, valid_mask)
        retained_mask = np.asarray(valid_mask, dtype=bool).copy()
        action_idx = sample_log_probs(execution_log_probs, rng)
    else:
        raise ValueError("mode must be 'greedy', 'topk', or 'numpy_policy'.")
    return action_idx, execution_log_probs, retained_mask


def entropy_bits_from_normalized_log_probs(log_probs: np.ndarray) -> np.ndarray:
    """Compute row-wise entropy in bits, safely ignoring zero-probability entries."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    probabilities = np.exp(log_probs)
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * log_probs[positive]
    return -np.sum(terms, axis=1) / LOG2


def selected_surprisal_bits(
    log_probs: np.ndarray,
    action_idx: np.ndarray,
) -> np.ndarray:
    """Return ``-log2 p(a)`` for one selected action per row."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    action_idx = np.asarray(action_idx, dtype=np.int64)
    selected = log_probs[np.arange(log_probs.shape[0]), action_idx]
    if np.any(~np.isfinite(selected)):
        raise ValueError("Selected actions must have finite log probability.")
    return -selected / LOG2


def accumulate_execution_path_log_probs(
    cumulative_log_probs: np.ndarray,
    execution_log_probs: np.ndarray,
    action_idx: np.ndarray,
) -> np.ndarray:
    """Add selected execution-policy log probabilities to a path score."""
    cumulative_log_probs = np.asarray(cumulative_log_probs, dtype=np.float64)
    execution_log_probs = np.asarray(execution_log_probs, dtype=np.float64)
    action_idx = np.asarray(action_idx, dtype=np.int64)
    if execution_log_probs.ndim != 2:
        raise ValueError("execution_log_probs must have shape [num_rollouts, num_actions].")
    if cumulative_log_probs.shape != (execution_log_probs.shape[0],):
        raise ValueError("cumulative_log_probs must contain one score per rollout.")
    if action_idx.shape != cumulative_log_probs.shape:
        raise ValueError("action_idx must contain one selected action per rollout.")
    if np.any(action_idx < 0) or np.any(action_idx >= execution_log_probs.shape[1]):
        raise ValueError("action_idx contains an out-of-range action.")
    selected = execution_log_probs[np.arange(execution_log_probs.shape[0]), action_idx]
    if np.any(~np.isfinite(selected)):
        raise ValueError("Selected execution-policy actions must have finite log probability.")
    return cumulative_log_probs + selected


def effective_fixed_rank_bits(effective_support: np.ndarray) -> np.ndarray:
    """Fixed-width local-rank bits for the retained action support."""
    effective_support = np.asarray(effective_support, dtype=np.int64)
    if np.any(effective_support < 1):
        raise ValueError("effective_support must be at least one.")
    return np.ceil(np.log2(effective_support)).astype(np.float64)


def shannon_integer_code_length(surprisal_bits: np.ndarray) -> np.ndarray:
    """Integer Shannon length ``ceil(-log2 p)``; a deterministic symbol costs zero."""
    surprisal_bits = np.asarray(surprisal_bits, dtype=np.float64)
    if np.any(~np.isfinite(surprisal_bits)) or np.any(surprisal_bits < -1e-12):
        raise ValueError("surprisal_bits must be finite and nonnegative.")
    return np.ceil(np.maximum(surprisal_bits, 0.0)).astype(np.float64)


def build_global_relation_prior(
    array_store: np.ndarray,
    num_relations: int,
    invalid_relation_ids: Iterable[int],
    alpha: float = 1.0,
) -> np.ndarray:
    """Build a smoothed relation prior from the capped graph action representation."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive so every allowed relation has support.")
    if num_relations < 1:
        raise ValueError("num_relations must be positive.")
    array_store = np.asarray(array_store)
    if array_store.ndim != 3 or array_store.shape[2] != 2:
        raise ValueError("array_store must have shape [num_entities, num_actions, 2].")

    invalid_ids = {int(value) for value in invalid_relation_ids}
    allowed = np.ones(num_relations, dtype=bool)
    for relation_id in invalid_ids:
        if 0 <= relation_id < num_relations:
            allowed[relation_id] = False

    relation_ids = array_store[:, :, 1].reshape(-1).astype(np.int64)
    in_range = (relation_ids >= 0) & (relation_ids < num_relations)
    observed = relation_ids[in_range & ~np.isin(relation_ids, list(invalid_ids))]
    counts = np.bincount(observed, minlength=num_relations).astype(np.float64)
    weights = np.where(allowed, counts + alpha, 0.0)
    if weights.sum() <= 0.0:
        raise ValueError("No allowed relations are available for the global prior.")
    return weights / weights.sum()


def task_agnostic_local_log_probs(
    next_relations: np.ndarray,
    valid_mask: np.ndarray,
    relation_prior: np.ndarray,
) -> np.ndarray:
    """Construct the local relation-frequency prior, splitting mass across duplicate actions."""
    next_relations = np.asarray(next_relations, dtype=np.int64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    relation_prior = np.asarray(relation_prior, dtype=np.float64)
    if next_relations.ndim != 2 or valid_mask.shape != next_relations.shape:
        raise ValueError("next_relations and valid_mask must have the same two-dimensional shape.")
    if np.any(valid_mask.sum(axis=1) == 0):
        raise ValueError("Every row must contain at least one valid action.")

    result = np.full(next_relations.shape, -np.inf, dtype=np.float64)
    for row in range(next_relations.shape[0]):
        local_relations = next_relations[row, valid_mask[row]]
        if np.any(local_relations < 0) or np.any(local_relations >= relation_prior.shape[0]):
            raise ValueError("A valid local relation is outside relation_prior.")
        unique_relations, multiplicities = np.unique(local_relations, return_counts=True)
        relation_weights = relation_prior[unique_relations]
        if np.any(relation_weights <= 0.0):
            raise ValueError("Every valid local relation must have nonzero prior probability.")
        relation_probabilities = relation_weights / relation_weights.sum()
        per_action = {
            int(relation_id): probability / multiplicity
            for relation_id, probability, multiplicity in zip(
                unique_relations, relation_probabilities, multiplicities
            )
        }
        action_probabilities = np.array(
            [per_action[int(relation_id)] for relation_id in local_relations], dtype=np.float64
        )
        result[row, valid_mask[row]] = np.log(action_probabilities)
    return result


def cross_entropy_bits(
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
) -> np.ndarray:
    """Compute row-wise cross entropy ``H(policy, reference)`` in bits."""
    policy_log_probs = np.asarray(policy_log_probs, dtype=np.float64)
    reference_log_probs = np.asarray(reference_log_probs, dtype=np.float64)
    if policy_log_probs.shape != reference_log_probs.shape or policy_log_probs.ndim != 2:
        raise ValueError("Policy and reference log probabilities must share a 2-D shape.")
    probabilities = np.exp(policy_log_probs)
    positive = probabilities > 0.0
    if np.any(~np.isfinite(reference_log_probs[positive])):
        raise ValueError("The reference must assign nonzero mass wherever the policy does.")
    terms = np.zeros_like(probabilities)
    terms[positive] = probabilities[positive] * reference_log_probs[positive]
    return -np.sum(terms, axis=1) / LOG2


def kl_bits(
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
) -> np.ndarray:
    """Compute row-wise ``D_KL(policy || reference)`` in bits."""
    cross_entropy = cross_entropy_bits(policy_log_probs, reference_log_probs)
    entropy = entropy_bits_from_normalized_log_probs(policy_log_probs)
    divergence = cross_entropy - entropy
    return np.maximum(divergence, 0.0)


def fixed_and_gold_hop_path_costs(
    per_step_cost: np.ndarray,
    gold_hops: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate the same trajectory over fixed horizon and a question-level gold-hop mask."""
    per_step_cost = np.asarray(per_step_cost, dtype=np.float64)
    gold_hops = np.asarray(gold_hops, dtype=np.int64)
    if per_step_cost.ndim != 3:
        raise ValueError("per_step_cost must have shape [questions, rollouts, horizon].")
    if gold_hops.shape != (per_step_cost.shape[0],):
        raise ValueError("gold_hops must contain one value per question.")
    if np.any(gold_hops < 0) or np.any(gold_hops > per_step_cost.shape[2]):
        raise ValueError("gold_hops must lie between zero and the fixed horizon.")
    if np.any(per_step_cost < -1e-12):
        raise ValueError("Path costs must be nonnegative.")

    fixed = per_step_cost.sum(axis=2)
    mask = np.arange(per_step_cost.shape[2])[None, None, :] < gold_hops[:, None, None]
    gold = np.where(mask, per_step_cost, 0.0).sum(axis=2)
    return fixed, gold


def question_total_cost_bits(per_step_cost: np.ndarray) -> np.ndarray:
    """Sum communication over every rollout and hop for each question.

    ``per_step_cost`` must use the protocol-explicit shape ``[B, R, T]``.
    The returned ``[B]`` values are ensemble totals, not per-path averages.
    """
    per_step_cost = np.asarray(per_step_cost, dtype=np.float64)
    if per_step_cost.ndim != 3:
        raise ValueError("per_step_cost must have shape [questions, rollouts, horizon].")
    if np.any(~np.isfinite(per_step_cost)):
        raise ValueError("Communication costs must be finite.")
    if np.any(per_step_cost < -1e-12):
        raise ValueError("Communication costs must be nonnegative.")
    return np.maximum(per_step_cost, 0.0).sum(axis=(1, 2))
