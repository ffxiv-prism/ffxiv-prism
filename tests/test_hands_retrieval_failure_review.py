from __future__ import annotations

import unittest

from scripts.build_hands_retrieval_failure_review import select_candidate_row


class HandsRetrievalFailureReviewTests(unittest.TestCase):
    def test_prefers_matching_gender_then_complete_candidate_crop(self) -> None:
        rows = [
            {"model_gender": "male", "candidate_crop": "both_hands", "image_row_id": 1},
            {"model_gender": "female", "candidate_crop": "left_hand", "image_row_id": 2},
            {"model_gender": "female", "candidate_crop": "both_hands", "image_row_id": 3},
        ]
        self.assertIs(select_candidate_row(rows, "female"), rows[2])


if __name__ == "__main__":
    unittest.main()
