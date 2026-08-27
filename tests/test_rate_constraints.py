import os
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))

from policy_entropy.rate_constraints import (  # noqa: E402
    build_global_relation_prior,
    cross_entropy_bits,
    effective_fixed_rank_bits,
    entropy_bits_from_normalized_log_probs,
    fixed_and_gold_hop_path_costs,
    kl_bits,
    normalized_valid_log_probs,
    select_actions_from_log_probs,
    select_greedy,
    selected_surprisal_bits,
    shannon_integer_code_length,
    task_agnostic_local_log_probs,
    topk_renormalized_log_probs,
    valid_action_mask,
)


class RateConstraintTests(unittest.TestCase):
    def setUp(self):
        self.log_probs = np.log(
            np.array(
                [
                    [0.50, 0.30, 0.20, 1e-30],
                    [0.10, 0.60, 0.30, 1e-30],
                ],
                dtype=np.float64,
            )
        )
        self.next_relations = np.array([[1, 2, 3, 0], [2, 2, 4, 0]])
        self.valid = valid_action_mask(self.next_relations, pad_relation_id=0)

    def test_valid_mask_rejects_all_pad_rows(self):
        with self.assertRaises(ValueError):
            valid_action_mask(np.zeros((1, 3), dtype=np.int32), pad_relation_id=0)

    def test_topk_retains_valid_support_and_normalizes(self):
        truncated, retained = topk_renormalized_log_probs(self.log_probs, self.valid, 2)
        np.testing.assert_array_equal(retained.sum(axis=1), np.array([2, 2]))
        self.assertFalse(np.any(retained[:, 3]))
        np.testing.assert_allclose(np.exp(truncated).sum(axis=1), 1.0, atol=1e-12)

    def test_topk_at_least_valid_count_matches_original_valid_distribution(self):
        truncated, retained = topk_renormalized_log_probs(self.log_probs, self.valid, 8)
        original = normalized_valid_log_probs(self.log_probs, self.valid)
        np.testing.assert_array_equal(retained, self.valid)
        np.testing.assert_allclose(truncated[self.valid], original[self.valid], atol=1e-12)

    def test_greedy_is_valid_and_uses_lowest_index_for_ties(self):
        tied = np.log(np.array([[0.4, 0.4, 0.2, 1e-30]]))
        valid = np.array([[True, True, True, False]])
        np.testing.assert_array_equal(select_greedy(tied, valid), np.array([0]))

    def test_k_one_matches_greedy_and_has_zero_rank_cost(self):
        greedy = select_greedy(self.log_probs, self.valid)
        sampled, execution, retained = select_actions_from_log_probs(
            self.log_probs,
            self.valid,
            mode="topk",
            top_k=1,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(sampled, greedy)
        np.testing.assert_array_equal(retained.sum(axis=1), np.ones(2, dtype=int))
        np.testing.assert_allclose(entropy_bits_from_normalized_log_probs(execution), 0.0)
        np.testing.assert_allclose(effective_fixed_rank_bits(retained.sum(axis=1)), 0.0)

    def test_seeded_topk_sampling_never_leaves_retained_support(self):
        sampled, _, retained = select_actions_from_log_probs(
            self.log_probs,
            self.valid,
            mode="topk",
            top_k=2,
            rng=np.random.default_rng(7),
        )
        self.assertTrue(np.all(retained[np.arange(sampled.shape[0]), sampled]))

    def test_fixed_rank_cost_never_exceeds_nominal_budget(self):
        for top_k in (1, 2, 4, 8):
            support = np.minimum(top_k, self.valid.sum(axis=1))
            costs = effective_fixed_rank_bits(support)
            self.assertTrue(np.all(costs <= np.log2(top_k)))

    def test_surprisal_and_integer_shannon_bounds(self):
        normalized = normalized_valid_log_probs(self.log_probs, self.valid)
        action_idx = np.array([0, 1])
        surprisal = selected_surprisal_bits(normalized, action_idx)
        lengths = shannon_integer_code_length(surprisal)
        self.assertTrue(np.all(np.isfinite(surprisal)))
        self.assertTrue(np.all(lengths == np.floor(lengths)))
        self.assertTrue(np.all(surprisal <= lengths))
        self.assertTrue(np.all(lengths < surprisal + 1.0))

    def test_graph_prior_and_local_duplicate_relation_split(self):
        array_store = np.array(
            [
                [[10, 1], [11, 2], [0, 0]],
                [[12, 2], [13, 2], [0, 0]],
            ],
            dtype=np.int32,
        )
        prior = build_global_relation_prior(array_store, 5, invalid_relation_ids=[0, 4])
        self.assertAlmostEqual(float(prior.sum()), 1.0)
        self.assertEqual(prior[0], 0.0)
        self.assertGreater(prior[3], 0.0)  # smoothing covers unobserved allowed relations

        local_relations = np.array([[2, 2, 1, 0]])
        local_valid = valid_action_mask(local_relations, pad_relation_id=0)
        local_log_probs = task_agnostic_local_log_probs(local_relations, local_valid, prior)
        local_probs = np.exp(local_log_probs)
        self.assertAlmostEqual(float(local_probs.sum()), 1.0)
        self.assertAlmostEqual(local_probs[0, 0], local_probs[0, 1])
        self.assertEqual(local_probs[0, 3], 0.0)

    def test_cross_entropy_and_kl_are_consistent(self):
        policy = normalized_valid_log_probs(self.log_probs, self.valid)
        reference = np.full(self.valid.shape, -np.inf, dtype=np.float64)
        uniform = np.broadcast_to(
            1.0 / self.valid.sum(axis=1, keepdims=True), self.valid.shape
        )
        reference[self.valid] = np.log(uniform[self.valid])
        cross_entropy = cross_entropy_bits(policy, reference)
        entropy = entropy_bits_from_normalized_log_probs(policy)
        np.testing.assert_allclose(kl_bits(policy, reference), cross_entropy - entropy)

    def test_gold_hop_cost_is_a_diagnostic_subset(self):
        costs = np.arange(1, 13, dtype=np.float64).reshape(2, 2, 3)
        fixed, gold = fixed_and_gold_hop_path_costs(costs, np.array([2, 3]))
        self.assertTrue(np.all(gold <= fixed))
        np.testing.assert_allclose(gold[1], fixed[1])


if __name__ == "__main__":
    unittest.main()
