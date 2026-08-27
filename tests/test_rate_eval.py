import os
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))

from policy_entropy.rate_eval import (  # noqa: E402
    QUESTION_TOTAL_SUMMARY_FIELDS,
    minerva_max_pool_metrics,
    recover_raw_action_counts,
    resolve_evaluation_overrides,
)


class MaxPoolMetricTests(unittest.TestCase):
    def test_duplicate_incorrect_entities_consume_one_rank(self):
        result = minerva_max_pool_metrics(
            final_entities=np.array([[10, 10, 20, 30]]),
            answer_hits=np.array([[False, False, True, False]]),
            path_log_probs=np.array([[-0.1, -0.2, -0.3, -0.4]]),
        )
        np.testing.assert_array_equal(result["answer_ranks"], np.array([1]))
        np.testing.assert_allclose(result["reciprocal_ranks"], np.array([0.5]))
        self.assertFalse(result["hits_at_1"][0])

    def test_multi_answer_hit_mask_accepts_first_gold_endpoint(self):
        result = minerva_max_pool_metrics(
            final_entities=np.array([[40, 20, 30]]),
            answer_hits=np.array([[True, False, True]]),
            path_log_probs=np.array([[-0.1, -0.2, -0.3]]),
        )
        self.assertEqual(result["answer_ranks"][0], 0)
        self.assertTrue(result["hits_at_1"][0])
        self.assertEqual(result["sorted_indices"][0, 0], 0)

    def test_missing_answer_has_zero_reciprocal_rank(self):
        result = minerva_max_pool_metrics(
            final_entities=np.array([[1, 2]]),
            answer_hits=np.array([[False, False]]),
            path_log_probs=np.array([[-0.2, -0.1]]),
        )
        self.assertEqual(result["answer_ranks"][0], -1)
        self.assertEqual(result["reciprocal_ranks"][0], 0.0)

    def test_r_one_hits_at_1_equals_rollout_success(self):
        answer_hits = np.array([[True], [False], [True]])
        result = minerva_max_pool_metrics(
            final_entities=np.array([[10], [20], [30]]),
            answer_hits=answer_hits,
            path_log_probs=np.array([[-0.1], [-0.2], [-0.3]]),
        )
        np.testing.assert_array_equal(result["hits_at_1"], answer_hits[:, 0])
        self.assertEqual(result["hits_at_1"].mean(), answer_hits.mean())

    def test_repeated_greedy_rollouts_have_mrr_equal_hits_at_1(self):
        answer_hits = np.array(
            [[True, True, True, True], [False, False, False, False]]
        )
        result = minerva_max_pool_metrics(
            final_entities=np.array([[10, 10, 10, 10], [20, 20, 20, 20]]),
            answer_hits=answer_hits,
            path_log_probs=np.zeros((2, 4)),
        )
        self.assertEqual(result["reciprocal_ranks"].mean(), result["hits_at_1"].mean())


class EvaluationOverrideTests(unittest.TestCase):
    def setUp(self):
        self.options = {
            "test_rollouts": 100,
            "max_num_actions": 200,
            "model_load_dir": "unchanged/model.ckpt",
        }

    def test_omitted_overrides_preserve_configured_values(self):
        effective, metadata = resolve_evaluation_overrides(self.options)
        self.assertEqual(effective["test_rollouts"], 100)
        self.assertEqual(effective["max_num_actions"], 200)
        self.assertEqual(metadata["configured_test_rollouts"], 100)
        self.assertEqual(metadata["effective_test_rollouts"], 100)
        self.assertFalse(metadata["test_rollouts_overridden"])

    def test_rollout_and_cap_overrides_are_metadata_visible_without_checkpoint_change(self):
        effective, metadata = resolve_evaluation_overrides(
            self.options, test_rollouts=1, max_num_actions=512
        )
        self.assertEqual(effective["test_rollouts"], 1)
        self.assertEqual(effective["max_num_actions"], 512)
        self.assertEqual(metadata["configured_test_rollouts"], 100)
        self.assertEqual(metadata["effective_test_rollouts"], 1)
        self.assertEqual(metadata["configured_max_num_actions"], 200)
        self.assertEqual(metadata["effective_max_num_actions"], 512)
        self.assertEqual(effective["model_load_dir"], self.options["model_load_dir"])
        self.assertTrue(metadata["model_load_dir_unchanged"])

    def test_multi_rollout_schema_requires_question_total_fields(self):
        expected = {
            "mean_question_fixed_rank_bits",
            "mean_question_surprisal_bits",
            "mean_question_shannon_code_bits",
            "mean_question_entropy_sum_bits",
        }
        self.assertTrue(expected.issubset(set(QUESTION_TOTAL_SUMMARY_FIELDS)))



class RawActionCountTests(unittest.TestCase):
    def test_reconstruction_respects_directed_inverse_filter_and_special_slot(self):
        class FakeGrapher:
            pass

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as graph_file:
            graph_file.write("a\tr\tb\n")
            graph_file.write("a\t_r\tc\n")
            graph_file.write("b\tr\ta\n")
            graph_file.flush()

            grapher = FakeGrapher()
            grapher.triple_store = graph_file.name
            grapher.entity_vocab = {"PAD": 0, "a": 1, "b": 2, "c": 3}
            grapher.relation_vocab = {"PAD": 0, "NO_OP": 1, "r": 2, "_r": 3}
            grapher.inverse_tokens = {3}
            grapher.use_directed_graph = True
            grapher.use_stop_signal = False
            grapher.use_restart_signal = False
            grapher.rPAD = 0
            # a: NO_OP+r, b: NO_OP+r, c: absent head, all exactly match raw counts.
            grapher.array_store = np.array(
                [
                    [[0, 0], [0, 0], [0, 0]],
                    [[1, 1], [2, 2], [0, 0]],
                    [[2, 1], [1, 2], [0, 0]],
                    [[0, 0], [0, 0], [0, 0]],
                ],
                dtype=np.int32,
            )

            counts, metadata = recover_raw_action_counts(grapher)
            np.testing.assert_array_equal(counts, np.array([0, 2, 2, 0]))
            self.assertTrue(metadata["available"])
            self.assertTrue(metadata["directed_inverse_filtering"])


if __name__ == "__main__":
    unittest.main()
