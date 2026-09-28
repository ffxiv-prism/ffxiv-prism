import unittest

import numpy as np

from ffxiv_prism.matching.local_fusion import (
    appearance_keys,
    fused_scores,
    ranks_of_truth,
    score_ranks,
)
from ffxiv_prism.matching.local_retrieval import (
    grouped_eye_ranks,
    rerank_candidates,
    search_ranks,
    visual_gates,
)


class LocalFusionTests(unittest.TestCase):
    def test_local_evidence_can_change_top_one_and_recall_a_missing_appearance(self):
        baseline = np.arange(1, 501, dtype=np.float32)
        baseline[-1] = 1_000_000
        local = np.broadcast_to(np.arange(2, 502, dtype=np.float32), (2, 2, 500)).copy()
        local[:, :, -1] = 1
        result = fused_scores(baseline, local, 1.0, weight=20.0, rrf_k=10)
        self.assertEqual(int(result.argmax()), 499)
        self.assertEqual(int(score_ranks(result)[-1]), 1)
        self.assertEqual(baseline[-1], 1_000_000)

    def test_no_visual_evidence_keeps_missing_candidates_missing(self):
        baseline = np.array([3.0, 1.0, 1_000_000, 2.0])
        local = np.ones((2, 2, 4))
        for gate, weight in ((0.0, 3.0), (1.0, 0.0)):
            result = fused_scores(baseline, local, gate, weight=weight, rrf_k=10)
            np.testing.assert_array_equal(score_ranks(result), baseline)
            self.assertTrue(np.isneginf(result[2]))

    def test_batched_and_single_query_fusion_agree(self):
        base = np.array([[1.0, 3.0, 2.0], [3.0, 1.0, 2.0]])
        local = np.array([[[[3.0, 1.0, 2.0], [2.0, 1.0, 3.0]]]] * 2)
        gates = np.array([0.2, 0.8])
        for mode in ("max", "mean"):
            batched = fused_scores(
                base, local, gates, weight=1.0, rrf_k=30, view_fusion=mode
            )
            for i in range(2):
                np.testing.assert_allclose(
                    batched[i],
                    fused_scores(
                        base[i],
                        local[i],
                        gates[i],
                        weight=1.0,
                        rrf_k=30,
                        view_fusion=mode,
                    ),
                )

    def test_equivalence_is_transitive_and_direction_independent(self):
        keys = appearance_keys({7: [9], 3: [9], 20: [21]})
        self.assertEqual([keys[i] for i in (3, 7, 9)], [3, 3, 3])
        self.assertEqual(keys[21], 20)

    def test_truth_scoring_handles_absent_labels_and_ties(self):
        result = ranks_of_truth(
            np.array([[1.0, 1.0, 0.0], [1.0, 0.0, -np.inf]]),
            np.array([[False, True, False], [False, False, True]]),
        )
        np.testing.assert_array_equal(result, [2, 1_000_000])

    def test_invalid_gates_and_shapes_are_rejected(self):
        for gate in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                fused_scores(np.ones(4), np.ones((2, 2, 4)), gate, weight=1, rrf_k=10)
        with self.assertRaises(ValueError):
            fused_scores(np.ones(4), np.ones((2, 2, 3)), 1.0, weight=1, rrf_k=10)

    def test_production_ranking_matches_benchmark_and_recalls_outside_global_pool(self):
        candidates = [
            dict(piece_id=10, rank=1, score=0.9),
            dict(piece_id=20, rank=2, score=0.8),
        ]
        full = [dict(local_path="third.png", pieces=[dict(piece_id=30)])]
        ids = [10, 20, 30]
        local = np.array([[[50.0, 60.0, 1.0], [50.0, 60.0, 1.0]]])
        policy = dict(weight=2.0, rrf_k=10, view_fusion="mean")
        output = rerank_candidates(candidates, full, ids, {}, local, 0.9, policy)
        expected = fused_scores(search_ranks(candidates, ids, {}), local, 0.9, **policy)
        self.assertEqual(
            [r["piece_id"] for r in output], [ids[i] for i in np.argsort(-expected)]
        )
        self.assertEqual(output[0]["piece_id"], 30)
        self.assertEqual(output[0]["local_preview"], "third.png")
        self.assertIsNone(output[0]["local_fusion"]["previous_rank"])
        self.assertEqual(candidates[0]["score"], 0.9)

    def test_candidate_views_are_grouped_before_ranking(self):
        # [query eyes, candidate renders, candidate eyes]; first two renders are equivalents.
        raw = np.array(
            [[[0.2, 0.3], [0.9, 0.5], [0.7, 0.1]], [[0.2, 0.1], [0.4, 0.3], [0.8, 0.9]]]
        )
        ranks = grouped_eye_ranks(raw, [[0, 1], [2], []])
        np.testing.assert_array_equal(ranks, [[1, 2, 1_000_000], [2, 1, 1_000_000]])
        np.testing.assert_array_equal(
            search_ranks([dict(piece_id=i) for i in (10, 11, 20)], [10, 20], {11: 10}),
            [1, 2],
        )

    def test_visual_gates_are_continuous_and_batch_consistent(self):
        for slot in ("HEAD", "LEGS"):
            p = np.array([[0.2, 0.3, 0.1, 0.4], [0.1, 0.2, 0.3, 0.4]])
            if slot == "HEAD":
                from ffxiv_prism.matching.head_detail import HEAD_TYPE_CLASSES

                p = np.full((2, len(HEAD_TYPE_CLASSES)), 0.2)
            single = visual_gates(slot, p)
            np.testing.assert_allclose(
                visual_gates(slot, np.stack([p, p])), [single, single]
            )
            self.assertTrue(np.all((single >= 0) & (single <= 1)))


if __name__ == "__main__":
    unittest.main()
