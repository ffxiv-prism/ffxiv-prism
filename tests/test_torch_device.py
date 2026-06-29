from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from ffxiv_prism.torch_device import resolve_torch_device


class TorchDeviceTests(unittest.TestCase):
    def test_auto_falls_back_to_cpu(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(resolve_torch_device("auto").type, "cpu")

    def test_auto_prefers_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_torch_device("auto").type, "cuda")

    def test_explicit_unavailable_cuda_has_actionable_error(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "--device auto"):
                resolve_torch_device("cuda")

    def test_accepts_existing_device(self) -> None:
        self.assertEqual(resolve_torch_device(torch.device("cpu")), torch.device("cpu"))
