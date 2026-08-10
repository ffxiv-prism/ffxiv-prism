from __future__ import annotations

import unittest

from scripts.audit_hands_candidate_routing import classify_route_coverage


class HandsCandidateRoutingAuditTests(unittest.TestCase):
    def test_classifies_strict_fallback_and_uncovered(self) -> None:
        self.assertEqual(
            classify_route_coverage(strict_indices=[1], any_gender_indices=[1]),
            "strict_covered",
        )
        self.assertEqual(
            classify_route_coverage(strict_indices=[], any_gender_indices=[2]),
            "requires_gender_fallback",
        )
        self.assertEqual(
            classify_route_coverage(strict_indices=[], any_gender_indices=[]),
            "route_uncovered",
        )


if __name__ == "__main__":
    unittest.main()
