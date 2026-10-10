import unittest
from unittest.mock import patch

import numpy as np

from ffxiv_prism.matching.runtime.encoders import similarity_order
from ffxiv_prism.matching.local_fusion import score_ranks
from ffxiv_prism import runtime_contract
from scripts.compare_inference_reports import compare
from ffxiv_prism.torch_device import configure_inference_precision


class DeviceParityTests(unittest.TestCase):
    def test_fp32_policy_also_disables_cudnn_tf32(self):
        import torch
        configure_inference_precision()
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "ieee")
        self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "ieee")

    def test_runtime_accepts_device_builds_but_rejects_missing_backend(self):
        info = {"packages": {**runtime_contract.VERSIONS, "torch": "2.11.0+cu130",
                             "torchvision": "0.26.0+cu130"}, "jpeg_turbo": "3.1.0"}
        with patch.object(runtime_contract, "runtime_info", return_value=info):
            runtime_contract.validate_runtime()
            info["packages"]["torchvision"] = None
            with self.assertRaisesRegex(RuntimeError, "torchvision"):
                runtime_contract.validate_runtime()

    def test_report_comparison_detects_same_set_in_different_order(self):
        a = {"device": "cpu", "slots": [{"slot": "HEAD", "candidates": [{"piece_id": 1}, {"piece_id": 2}]}]}
        b = {"device": "cuda", "slots": [{"slot": "HEAD", "candidates": [{"piece_id": 2}, {"piece_id": 1}]}]}
        self.assertFalse(compare([a, b])["ok"])

    def test_float32_noise_and_exact_duplicates_use_catalogue_order(self):
        cpu = np.array([.60000002, .60000008, .8, .6, .3], dtype=np.float32)
        cuda = np.array([.60000008, .60000002, .8, .6, .3], dtype=np.float32)
        np.testing.assert_array_equal(similarity_order(cpu), [2, 0, 1, 3, 4])
        np.testing.assert_array_equal(similarity_order(cpu), similarity_order(cuda))
        np.testing.assert_array_equal(score_ranks(cpu), score_ranks(cuda))

    def test_distinguishable_scores_and_unavailable_candidates_keep_order(self):
        scores = np.array([.60001, .6, -np.inf, .8], dtype=np.float32)
        np.testing.assert_array_equal(similarity_order(scores), [3, 0, 1, 2])
        np.testing.assert_array_equal(score_ranks(scores), [2, 3, 1000000, 1])
