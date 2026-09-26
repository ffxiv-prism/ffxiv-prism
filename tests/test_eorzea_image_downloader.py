from __future__ import annotations

import unittest

from scripts.download_eorzea_images_from_url_manifest import (
    browser_payload_needs_verification,
    select_manifest_rows_to_append,
)
from scripts.build_eorzea_image_url_manifest_from_list_cache import (
    build_manifest_rows,
    browser_retryable_check,
    retained_manifest_urls,
)


class EorzeaImageDownloaderTests(unittest.TestCase):
    def test_html_403_is_retried_through_browser(self) -> None:
        payload = {"status": 403, "content_type": "text/html; charset=UTF-8"}

        self.assertTrue(browser_retryable_check(payload))
        self.assertTrue(browser_payload_needs_verification(payload))

    def test_image_404_is_not_treated_as_verification(self) -> None:
        payload = {"status": 404, "content_type": "text/html"}

        self.assertFalse(browser_retryable_check(payload))
        self.assertFalse(browser_payload_needs_verification(payload))

    def test_unchanged_cache_hits_are_not_appended_again(self) -> None:
        existing = [{"source_image_url": "https://example.test/known.png"}]
        downloaded = [
            {
                "source_image_url": "https://example.test/known.png",
                "from_cache": True,
            },
            {
                "source_image_url": "https://example.test/new.png",
                "from_cache": True,
            },
            {
                "source_image_url": "https://example.test/refreshed.png",
                "from_cache": False,
            },
        ]

        selected = select_manifest_rows_to_append(existing, downloaded)

        self.assertEqual(
            [row["source_image_url"] for row in selected],
            [
                "https://example.test/new.png",
                "https://example.test/refreshed.png",
            ],
        )

    def test_resumed_manifest_preserves_checked_urls_missing_from_current_pages(self) -> None:
        current = {
            "https://gearsets.eorzeacollection.com/raid/current/current/hyur-female-body.png"
        }
        historical_url = (
            "https://gearsets.eorzeacollection.com/pvp/series/archfiend/"
            "hyur-female-feet.png"
        )
        historical = {
            "source_image_url": historical_url,
            "accessible": True,
            "image_variant": "normal",
        }

        scope = retained_manifest_urls(
            current,
            [historical],
            preserve_cached=True,
        )
        rows = build_manifest_rows([historical], scope)

        self.assertIn(historical_url, scope)
        self.assertEqual([row["source_image_url"] for row in rows], [historical_url])

    def test_reset_manifest_does_not_preserve_historical_scope(self) -> None:
        current = {
            "https://gearsets.eorzeacollection.com/raid/current/current/hyur-female-body.png"
        }
        historical = {
            "source_image_url": (
                "https://gearsets.eorzeacollection.com/pvp/series/archfiend/"
                "hyur-female-feet.png"
            )
        }

        scope = retained_manifest_urls(
            current,
            [historical],
            preserve_cached=False,
        )

        self.assertEqual(scope, current)


if __name__ == "__main__":
    unittest.main()
