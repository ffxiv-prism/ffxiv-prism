from __future__ import annotations

import unittest
from pathlib import Path

from scripts.update_active_candidate_indexes import Target, build_additions


class UpdateActiveCandidateIndexesTests(unittest.TestCase):
    def test_build_additions_expands_only_new_expected_gearset(self) -> None:
        target = Target(
            name="feet",
            source=Path("unused"),
            output=Path("unused-output"),
            model_id="unused",
            encoder_group="dinov2",
            slots=("FEET",),
            expected_new_rows=5,
            feet_crop_policy="multiscale",
        )
        base_rows = [
            {
                "image_row_id": 10,
                "source_image_url": "https://example.test/new-feet.png",
                "local_path": "data/new-feet.png",
                "slot": "FEET",
                "pieces": [{"gearset_slug": "neo-queens"}],
            }
        ]

        # Avoid filesystem fixture setup by verifying the pure expansion inputs
        # through a temporary source metadata shim.
        import scripts.update_active_candidate_indexes as module

        original_read_jsonl = module.read_jsonl
        module.read_jsonl = lambda _path: []
        try:
            additions = build_additions(
                target,
                base_rows,
                expected_gearset_slug="neo-queens",
            )
        finally:
            module.read_jsonl = original_read_jsonl

        self.assertEqual(len(additions), 5)
        self.assertEqual(
            {row["candidate_crop"] for row in additions},
            {
                "full",
                "feet_left",
                "feet_right",
                "feet_boot_left",
                "feet_boot_right",
            },
        )

    def test_build_additions_accepts_multiple_audited_gearsets(self) -> None:
        target = Target(
            name="body",
            source=Path("unused"),
            output=Path("unused-output"),
            model_id="unused",
            encoder_group="dinov2",
            slots=("BODY",),
            expected_new_rows=None,
        )
        base_rows = [
            {
                "source_image_url": f"https://example.test/{slug}.png",
                "local_path": f"data/{slug}.png",
                "slot": "BODY",
                "pieces": [{"gearset_slug": slug}],
            }
            for slug in ("first-set", "second-set")
        ]

        import scripts.update_active_candidate_indexes as module

        original_read_jsonl = module.read_jsonl
        module.read_jsonl = lambda _path: []
        try:
            additions = build_additions(
                target,
                base_rows,
                expected_gearset_slugs=frozenset(
                    ("first-set", "second-set")
                ),
            )
        finally:
            module.read_jsonl = original_read_jsonl

        self.assertEqual(len(additions), 2)


if __name__ == "__main__":
    unittest.main()
