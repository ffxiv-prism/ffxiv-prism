from __future__ import annotations

import unittest

from scripts.select_slot_quality_recovery_review_set import select_review_set


def row(example_id, glamour_id, slots, identities, race="Hyur", quality=0.9):
    return {
        "example_id": example_id,
        "glamour_id": glamour_id,
        "model_race": race,
        "quality_score": quality,
        "single_to_multi_conversion_slots": slots,
        "slot_details": {
            slot: {"appearance_identity": identity}
            for slot, identity in zip(slots, identities, strict=True)
        },
    }


class SelectSlotQualityRecoveryReviewSetTests(unittest.TestCase):
    def test_prefers_one_image_covering_multiple_slot_targets(self) -> None:
        rows = [
            row("a:0", "a", ["HEAD", "BODY"], [1, 2]),
            row("b:0", "b", ["HEAD"], [3]),
            row("c:0", "c", ["BODY"], [4]),
        ]

        selected = select_review_set(
            rows,
            targets={"HEAD": 1, "BODY": 1},
            require_roegadyn_head=False,
        )

        self.assertEqual([item["example_id"] for item in selected], ["a:0"])

    def test_requires_distinct_identities_not_duplicate_views(self) -> None:
        rows = [
            row("a:0", "a", ["HEAD"], [1]),
            row("b:0", "b", ["HEAD"], [1]),
            row("c:0", "c", ["HEAD"], [2]),
        ]

        selected = select_review_set(
            rows,
            targets={"HEAD": 2},
            require_roegadyn_head=False,
        )

        self.assertEqual(
            {item["slot_details"]["HEAD"]["appearance_identity"] for item in selected},
            {1, 2},
        )

    def test_roegadyn_head_requirement_is_enforced(self) -> None:
        rows = [
            row("a:0", "a", ["HEAD"], [1], race="Hyur", quality=1.0),
            row("b:0", "b", ["HEAD"], [2], race="Roegadyn", quality=0.9),
        ]

        selected = select_review_set(rows, targets={"HEAD": 1})

        self.assertTrue(any(item["model_race"] == "Roegadyn" for item in selected))


if __name__ == "__main__":
    unittest.main()
