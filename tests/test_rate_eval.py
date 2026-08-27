import os
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))

from policy_entropy.rate_eval import (  # noqa: E402
    minerva_max_pool_metrics,
    recover_raw_action_counts,
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
