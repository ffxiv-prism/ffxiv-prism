from __future__ import annotations

import unittest

from scripts.build_source_domain_review_queue import domain_risk, domain_sort_key


class BuildSourceDomainReviewQueueTests(unittest.TestCase):
    def test_domain_risk_is_complement_of_game_probability(self) -> None:
        self.assertAlmostEqual(domain_risk({"ffxiv_game": 0.8}), 0.2)

    def test_sort_key_prioritizes_higher_risk(self) -> None:
        high = {"example_id": "high", "non_game_risk": 0.9}
        low = {"example_id": "low", "non_game_risk": 0.1}
        self.assertLess(domain_sort_key(high), domain_sort_key(low))


if __name__ == "__main__":
    unittest.main()
