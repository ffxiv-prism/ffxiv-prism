from __future__ import annotations

import unittest

from ffxiv_prism.matching.hands_morphology import hands_morphology_class


class HandsMorphologyTests(unittest.TestCase):
    def test_name_taxonomy(self) -> None:
        self.assertEqual(hands_morphology_class("False Nails"), "minimal")
        self.assertEqual(hands_morphology_class("Crystarium Gauntlets"), "long_armor")
        self.assertEqual(hands_morphology_class("Fingerless Gloves"), "glove")


if __name__ == "__main__":
    unittest.main()
