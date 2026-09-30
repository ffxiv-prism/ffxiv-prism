import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ffxiv_prism.matching.aligned_eyes import boxes_for_pose
from ffxiv_prism.matching.eye_accessory import encode_candidate_eyes
from ffxiv_prism.matching.eye_experts import (
    classify_eye_coverage, eye_expert_scores, grouped_coverage_profiles,
)
from ffxiv_prism.matching.inference import current_best_configs, base_configs
from ffxiv_prism.matching.local_fusion import fused_scores
from ffxiv_prism.matching.local_retrieval import rerank_candidates
from ffxiv_prism.vision.outfit_regions import Keypoint, build_eye_regions


class EyeExpertTests(unittest.TestCase):
    def fixture(self):
        query = np.array([[.8, 0, 0, 0, 0, .2], [.6, 0, 0, 0, 0, .4]])
        candidates = np.eye(6)[[5, 0, 0]]
        local = np.array([[[1, 9, 20], [1, 30, 10]], [[1, 6, 9], [1, 50, 7]]], dtype=float)
        return local, query, candidates

    def test_family_ranks_ignore_bare_faces_and_mix_two_models(self):
        local, q, c = self.fixture()
        scores = eye_expert_scores(local, q, c, 1.)
        np.testing.assert_allclose(scores, [0, .6/11, .6/11])
        # Unsupported appearances cannot contribute through their global rank.
        self.assertEqual(scores[0], 0)
        # No output position is reserved: strong image evidence can move first.
        policy = dict(weight=.5, rrf_k=10, view_fusion="max")
        rows = [dict(piece_id=i, rank=i+1, score=.1) for i in range(3)]
        baseline = fused_scores(np.arange(1, 4), local, 1., **policy)
        result = rerank_candidates(rows, [], [0, 1, 2], {}, local, 1., policy,
                                   extra_scores=scores)
        self.assertEqual([r["piece_id"] for r in result], np.argsort(-(baseline+scores)).tolist())
        self.assertNotEqual(result[0]["piece_id"], 0)

    def test_missing_views_and_bare_eyes_cannot_trigger_experts(self):
        local, q, c = self.fixture()
        for query, gate in ((q, 0.), (np.eye(6)[[5, 5]], 1.),
                            (np.eye(6)[[0, 1]], 1.)):
            np.testing.assert_array_equal(eye_expert_scores(local, query, c, gate), 0)
        local[:] = 1_000_000
        np.testing.assert_array_equal(eye_expert_scores(local, q, c, 1.), 0)

    def test_batch_inference_matches_individual_inference(self):
        local, q, c = self.fixture()
        batch = eye_expert_scores(np.stack([local, local]), np.stack([q, q]), c,
                                 np.array([.2, .9]))
        for i, gate in enumerate((.2, .9)):
            np.testing.assert_allclose(batch[i], eye_expert_scores(local, q, c, gate))

    def test_group_profiles_require_render_consensus_not_one_accidental_match(self):
        values = np.eye(6)[[0, 5, 0]]
        grouped = grouped_coverage_profiles(values, [[0, 1, 1], [2], []])
        np.testing.assert_array_equal(grouped[0], [.5, 0, 0, 0, 0, .5])
        np.testing.assert_array_equal(grouped[1], [1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(grouped[2], 0)

    def test_scores_follow_appearance_permutations_without_identity_features(self):
        local, q, c = self.fixture()
        permutation = [2, 0, 1]
        expected = eye_expert_scores(local, q, c, .8)[permutation]
        actual = eye_expert_scores(local[..., permutation], q, c[permutation], .8)
        np.testing.assert_allclose(expected, actual)

    def test_invalid_inputs_are_rejected(self):
        local, q, c = self.fixture()
        for bad_gate in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                eye_expert_scores(local, q, c, bad_gate)
        with self.assertRaises(ValueError):
            eye_expert_scores(local[:1], q, c, 1.)
        with self.assertRaises(ValueError):
            eye_expert_scores(local, q, c, 1., weight=float("nan"))

    def test_candidate_and_query_frames_match_with_pairwise_fallback(self):
        points = [Keypoint(x, y, .9) for x, y in
                  ((100, 100), (90, 90), (110, 90), (70, 100), (130, 100))]
        result = boxes_for_pose(points, (200, 200))
        self.assertTrue(result["aligned"])
        self.assertEqual(result["boxes"], [r.box for r in build_eye_regions(points, (200, 200))])
        points[1] = Keypoint(90, 90, .1)
        result = boxes_for_pose(points, (200, 200))
        self.assertFalse(result["aligned"])
        self.assertEqual(result["boxes"], [(56, 72, 106, 144), (94, 72, 144, 144)])

    def test_pixel_boxes_and_preprocessing_invalidate_embedding_cache(self):
        class Encoder:
            model_id = "test"
            preprocess = "color"
            calls = 0

            def prepare_image(self, path):
                with Image.open(path) as im:
                    return im.convert("RGB")

            def encode_paths(self, paths, batch_size):
                self.calls += 1
                result = []
                for path in paths:
                    with self.prepare_image(path) as im:
                        result.append(np.asarray(im).mean((0, 1)))
                return np.asarray(result)

        with tempfile.TemporaryDirectory() as folder, patch(
                "ffxiv_prism.matching.eye_accessory.model_weight_files", return_value=[]):
            root = Path(folder)
            path = root / "image.png"
            im = Image.new("RGB", (20, 20), "red")
            im.paste(Image.new("RGB", (10, 20), "blue"), (10, 0))
            im.save(path)
            encoder = Encoder()
            boxes = {str(path): [(0, 0, 10, 20), (10, 0, 20, 20)]}
            options = dict(batch_size=2, cache_dir=root / "cache", boxes_by_path=boxes)
            first = encode_candidate_eyes(encoder, [path], **options)
            np.testing.assert_array_equal(first[0], [[255, 0, 0], [0, 0, 255]])
            encode_candidate_eyes(encoder, [path], **options)
            self.assertEqual(encoder.calls, 2)
            boxes[str(path)].reverse()
            second = encode_candidate_eyes(encoder, [path], **options)
            np.testing.assert_array_equal(second, first[:, ::-1])
            self.assertEqual(encoder.calls, 4)
            encoder.preprocess = "gray"
            encode_candidate_eyes(encoder, [path], **options)
            self.assertEqual(encoder.calls, 6)

    def test_coverage_cache_tracks_content_and_recovers_from_corruption(self):
        calls = []

        def classify(model_id, prompts, paths, **kwargs):
            calls.append(len(paths))
            return np.tile(np.eye(6)[0], (len(paths), 1))

        with tempfile.TemporaryDirectory() as folder, patch(
                "ffxiv_prism.matching.eye_experts.asset_signature", return_value=[]):
            root = Path(folder)
            path = root / "image.png"
            Image.new("RGB", (20, 20), "red").save(path)
            kwargs = dict(classifier=classify, device="cpu", batch_size=2, cache_dir=root / "cache")
            classify_eye_coverage([path, path], **kwargs)
            classify_eye_coverage([path], **kwargs)
            self.assertEqual(calls, [1])
            next((root / "cache").glob("*.npy")).write_bytes(b"broken cache")
            classify_eye_coverage([path], **kwargs)
            Image.new("RGB", (20, 20), "blue").save(path)
            classify_eye_coverage([path], **kwargs)
            self.assertEqual(calls, [1, 1, 1])

    def test_only_current_best_head_enables_experts(self):
        self.assertEqual([s for s, c in current_best_configs().items() if c.head_eye_experts], ["HEAD"])
        self.assertFalse(any(c.head_eye_experts for c in base_configs().values()))


if __name__ == "__main__":
    unittest.main()
