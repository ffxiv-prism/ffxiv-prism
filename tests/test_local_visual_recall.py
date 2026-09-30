import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ffxiv_prism.matching.eye_accessory import (
    EYE_NAMES,
    PRESENCE_NAMES,
    encode_candidate_eyes,
)
from ffxiv_prism.matching.local_retrieval import local_records
from ffxiv_prism.matching.inference import (
    crop_weight,
    current_best_configs,
    base_configs,
)
from ffxiv_prism.vision.outfit_regions import Keypoint, build_eye_regions


class LocalVisualRecallTests(unittest.TestCase):
    def test_unreliable_views_disable_local_fusion(self):
        records = [
            dict(name=n, visibility="visible", confidence=0.9)
            for n in (*PRESENCE_NAMES, *EYE_NAMES)
        ]
        self.assertEqual(len(local_records(records, "HEAD")), 4)
        for bad_field in (
            dict(visibility="partially_visible"),
            dict(confidence=0.1),
            dict(tags=["source_edge_truncation_risk"]),
        ):
            bad_records = copy.deepcopy(records)
            bad_records[2].update(bad_field)
            self.assertEqual(local_records(bad_records, "HEAD"), [])

    def test_eye_crops_keep_uncertain_landmarks_out_of_retrieval(self):
        points = [
            Keypoint(x, y, 0.9)
            for x, y in ((100, 100), (90, 90), (110, 90), (70, 100), (130, 100))
        ]
        regions = build_eye_regions(points, (200, 200))
        self.assertTrue(all(r.visibility == "visible" for r in regions))
        self.assertTrue(all(crop_weight(vars(r)) == 0 for r in regions))
        points[1] = points[2]
        self.assertTrue(
            all(
                r.visibility == "pose_uncertain"
                for r in build_eye_regions(points, (200, 200))
            )
        )

    def test_candidate_cache_tracks_image_content_and_reuses_vectors(self):
        class Encoder:
            model_id = "test"
            calls = 0

            def prepare_image(self, path):
                with Image.open(path) as im:
                    return im.convert("RGB")

            def encode_paths(self, paths, batch_size):
                self.calls += 1
                vectors = []
                for path in paths:
                    with self.prepare_image(path) as im:
                        vectors.append(np.asarray(im).mean(axis=(0, 1)))
                return np.asarray(vectors)

        with (
            tempfile.TemporaryDirectory() as folder,
            patch(
                "ffxiv_prism.matching.eye_accessory.asset_signature", return_value=[]
            ),
        ):
            root = Path(folder)
            path = root / "image.png"
            Image.new("RGB", (40, 40), "red").save(path)
            encoder = Encoder()
            first = encode_candidate_eyes(
                encoder, [path], batch_size=2, cache_dir=root / "cache"
            )
            np.testing.assert_equal(
                first,
                encode_candidate_eyes(
                    encoder, [path], batch_size=2, cache_dir=root / "cache"
                ),
            )
            self.assertEqual(encoder.calls, 2)
            Image.new("RGB", (40, 40), "blue").save(path)
            second = encode_candidate_eyes(
                encoder, [path], batch_size=2, cache_dir=root / "cache"
            )
            self.assertFalse(np.array_equal(first, second))
            self.assertEqual(encoder.calls, 4)

    def test_base_profile_remains_without_local_recall(self):
        self.assertTrue(current_best_configs()["HEAD"].local_visual_fusion)
        self.assertTrue(current_best_configs()["LEGS"].local_visual_fusion)
        self.assertTrue(all(not c.local_visual_fusion for c in base_configs().values()))


if __name__ == "__main__":
    unittest.main()
