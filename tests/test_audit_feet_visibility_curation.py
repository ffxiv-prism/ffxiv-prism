from __future__ import annotations

import unittest

from scripts.audit_feet_visibility_curation import audit_feet_curation


class AuditFeetVisibilityCurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = [
            {"example_id": "a:0", "glamour_id": "a", "triage_for_review": True},
            {"example_id": "b:0", "glamour_id": "b", "triage_for_review": False},
        ]
        self.queue = [self.features[0]]
        self.reviews = [
            {"example_id": "a:0", "slot": "FEET", "usability": "unusable"},
            {"example_id": "b:0", "slot": "FEET", "usability": "usable"},
        ]
        self.negative = [self.features[1]]
        self.training = [
            {
                "example_id": "a:0",
                "decision": "accept",
                "slot_usability": {"FEET": "unusable"},
            }
        ]

    def audit(self, **overrides):
        return audit_feet_curation(
            overrides.get("features", self.features),
            overrides.get("queue", self.queue),
            overrides.get("reviews", self.reviews),
            overrides.get("negative", self.negative),
            overrides.get("training", self.training),
            min_negative_sample=1,
            min_usable_rate=0.95,
            min_wilson_low=0.0,
        )

    def test_ready_when_queue_review_sample_and_exclusion_are_complete(self) -> None:
        report = self.audit()
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["curation"]["known_unusable_images"], 1)
        self.assertEqual(report["curation"]["curated_images"], 1)

    def test_missing_queue_review_fails_closed(self) -> None:
        report = self.audit(reviews=[self.reviews[1]])
        self.assertEqual(report["status"], "do_not_train")
        self.assertFalse(report["checks"]["triage_queue_fully_reviewed"])

    def test_unapplied_unusable_review_fails_closed(self) -> None:
        report = self.audit(training=[])
        self.assertEqual(report["status"], "do_not_train")
        self.assertFalse(report["checks"]["known_unusable_exclusions_applied"])


if __name__ == "__main__":
    unittest.main()
