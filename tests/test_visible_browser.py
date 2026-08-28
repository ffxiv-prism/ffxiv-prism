from __future__ import annotations

import unittest
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.collectors.eorzea_collection.visible_browser import (
    normalized_page_url,
    page_has_cloudflare_challenge,
)
from scripts.discover_eorzea_gearsets_by_model import unseen_links


class VisibleBrowserTests(unittest.TestCase):
    def test_detects_cloudflare_challenge(self) -> None:
        html = "<title>Just a moment...</title><div>Cloudflare</div><div id='cf-chl-widget'></div>"
        self.assertTrue(page_has_cloudflare_challenge(html))

    def test_accepts_normal_gearset_page(self) -> None:
        html = "<html><title>Gearsets | Eorzea Collection</title><a href='/gearset/example'>Example</a></html>"
        self.assertFalse(page_has_cloudflare_challenge(html))

    def test_normalized_page_url_drops_cloudflare_token(self) -> None:
        self.assertEqual(
            normalized_page_url(
                "https://example.test/list?page=1&__cf_chl_rt_tk=secret"
            ),
            "https://example.test/list?page=1",
        )

    def test_repeated_list_page_has_no_unseen_links(self) -> None:
        seen = {"https://example.test/gearset/a", "https://example.test/gearset/b"}
        self.assertEqual(unseen_links(sorted(seen), seen), [])


if __name__ == "__main__":
    unittest.main()
