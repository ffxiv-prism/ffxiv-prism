from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ffxiv_prism.model_assets import is_remote_model_id, load_pretrained
from scripts.download_eorzea_images_from_url_manifest import (
    current_best_image_coverage,
    required_image_paths_from_metadata,
)
from scripts.export_current_best_assets import inference_checkpoint_payload


class CurrentBestAssetsTests(unittest.TestCase):
    def test_auto_processor_pins_backend_in_local_and_remote_loads(self) -> None:
        class AutoProcessor:
            calls = []

            @classmethod
            def from_pretrained(cls, _source, **kwargs):
                cls.calls.append(kwargs)
                if kwargs.get("local_files_only"):
                    raise OSError("not cached")
                return object()

        load_pretrained(AutoProcessor, "example/model")
        self.assertEqual([c["backend"] for c in AutoProcessor.calls],
                         ["torchvision", "torchvision"])

    def test_remote_model_ids_do_not_include_project_paths(self) -> None:
        self.assertTrue(is_remote_model_id("facebook/dinov2-base"))
        self.assertFalse(
            is_remote_model_id("var/models/dinov3-vitb16-pretrain-lvd1689m")
        )
        self.assertFalse(is_remote_model_id("data/models/current-best/model.pt"))

    def test_remote_model_load_retries_without_local_only(self) -> None:
        class Factory:
            calls: list[dict[str, object]] = []

            @classmethod
            def from_pretrained(cls, _model_id: str, **kwargs: object) -> object:
                cls.calls.append(kwargs)
                if kwargs.get("local_files_only"):
                    raise OSError("not cached")
                return object()

        result = load_pretrained(Factory, "example/model")

        self.assertIsNotNone(result)
        self.assertEqual(len(Factory.calls), 2)
        self.assertTrue(Factory.calls[0]["local_files_only"])
        self.assertNotIn("local_files_only", Factory.calls[1])

    def test_inference_checkpoint_drops_optimizer_and_history(self) -> None:
        payload = inference_checkpoint_payload(
            {
                "version": 1,
                "model_id": "example/model",
                "trainable_state_dict": {"weight": 1},
                "pooler_state_dict": {"weight": 2},
                "optimizer_state_dict": {"state": "large"},
                "history": [{"epoch": 1}],
            }
        )

        self.assertIn("trainable_state_dict", payload)
        self.assertIn("pooler_state_dict", payload)
        self.assertNotIn("optimizer_state_dict", payload)
        self.assertNotIn("history", payload)

    def test_required_image_paths_deduplicate_source_urls(self) -> None:
        rows = [
            {
                "source_image_url": "https://example.invalid/a.png",
                "local_path": "data/raw/eorzea_collection/images/a.png",
            },
            {
                "source_image_url": "https://example.invalid/a.png",
                "local_path": "data/raw/eorzea_collection/images/a.png",
            },
        ]

        required = required_image_paths_from_metadata(rows)

        self.assertEqual(len(required), 1)
        self.assertTrue(
            next(iter(required.values())).as_posix().endswith(
                "data/raw/eorzea_collection/images/a.png"
            )
        )

    def test_image_coverage_reports_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            available = root / "available.png"
            missing = root / "missing.png"
            available.write_bytes(b"image")

            report = current_best_image_coverage(
                {
                    "https://example.invalid/available.png": available,
                    "https://example.invalid/missing.png": missing,
                }
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["available_image_count"], 1)
        self.assertEqual(report["missing_image_count"], 1)


if __name__ == "__main__":
    unittest.main()
