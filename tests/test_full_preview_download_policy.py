from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ffxiv_prism.collectors.eorzea_collection.collector import EorzeaCollectionCollector
from ffxiv_prism.collectors.eorzea_collection.image_url import is_full_preview_image, parse_gearsets_image_url
from ffxiv_prism.collectors.eorzea_collection.models import ImageCandidate
from ffxiv_prism.dataset.update_audit import META_PATH, image_snapshot, update_result
from scripts import download_eorzea_images_from_url_manifest as downloader
from scripts import run_lifecycle_pipeline as lifecycle

BASE = 'https://gearsets.eorzeacollection.com/artifact/nin/momochi/hyur-female-'

def candidate(part):
    # Sparse metadata must not let legacy front/back URLs escape the filter.
    return ImageCandidate(source_image_url=BASE + part + '.png',
                          source_page_url='https://ffxiv.eorzeacollection.com/gearset/momochi',
                          image_kind='gearset_static')

class FullPreviewDownloadPolicyTests(unittest.TestCase):
    def collector(self, **kwargs):
        with patch('ffxiv_prism.collectors.eorzea_collection.collector.ensure_project_dirs'), patch('ffxiv_prism.collectors.eorzea_collection.collector.PoliteHttpClient'):
            return EorzeaCollectionCollector(**kwargs)

    def test_url_classification_precedes_download_and_keeps_slot_views(self):
        for part in ('front', 'back', 'dyed-front', 'dyed-back'):
            with self.subTest(part=part):
                self.assertTrue(is_full_preview_image(BASE + part + '.png'))
                self.assertIsNone(parse_gearsets_image_url(BASE + part + '.png').slot)
        for part in ('head', 'body', 'hands', 'legs', 'feet'):
            self.assertFalse(is_full_preview_image(BASE + part + '.png'))
            self.assertEqual(parse_gearsets_image_url(BASE + part + '.png').slot, part.upper())
        self.assertTrue(is_full_preview_image('https://example.test/preview.png', 'front'))
        self.assertFalse(is_full_preview_image('https://example.test/unknown.png'))

    def test_http_download_excludes_previews_before_sequential_or_parallel_requests(self):
        candidates = [candidate('front'), candidate('back'), candidate('body')]
        for workers in (1, 2):
            for include in (False, True):
                with self.subTest(workers=workers, include=include):
                    collector = self.collector(image_workers=workers, include_full_previews=include)
                    with patch.object(collector, '_download_image_record', return_value=None) as fetch:
                        collector.download_images(candidates)
                    self.assertCountEqual([call.args[0].source_image_url for call in fetch.call_args_list],
                                          [row.source_image_url for row in (candidates if include else candidates[-1:])])

    def test_browser_download_uses_the_same_policy_even_for_cached_images(self):
        candidates = [candidate('front'), candidate('back'), candidate('body')]
        with tempfile.TemporaryDirectory() as directory:
            cached = Path(directory) / 'cached.png'
            cached.write_bytes(b'cached')
            for include in (False, True):
                collector = self.collector(include_full_previews=include)
                with patch.object(collector, 'image_target', return_value=cached), patch.object(collector, 'image_is_readable', return_value=True), patch.object(collector, 'downloaded_image_record', side_effect=lambda row, *args: row), patch.object(downloader, 'create_visible_driver') as browser:
                    records = downloader.download_images_browser(collector, candidates, profile_dir=Path(directory), wait_seconds=1, progress=downloader.ImageDownloadProgress(3))
                browser.assert_not_called()
                self.assertEqual(records, candidates if include else candidates[-1:])

    def test_cli_defaults_opt_in_and_explicit_active_index_restoration(self):
        rows = [dict(source_image_url=BASE+part+'.png', source_page_url='https://ffxiv.eorzeacollection.com/gearset/momochi', image_kind='gearset_static', accessible=True) for part in ('front', 'back', 'body')]
        cases = [([], ['body']), (['--include-full-previews'], ['front', 'back', 'body']),
                 (['--current-best-only'], ['front']), (['--limit', '1'], ['body'])]
        for options, expected in cases:
            with self.subTest(options=options), redirect_stdout(io.StringIO()), patch('sys.argv', ['download', *options]), patch.object(downloader, 'read_jsonl', side_effect=[rows, []]), patch.object(downloader, 'append_jsonl'), patch.object(downloader, 'current_best_required_images', return_value={BASE+'front.png': Path('front.png')}), patch('ffxiv_prism.collectors.eorzea_collection.collector.ensure_project_dirs'), patch('ffxiv_prism.collectors.eorzea_collection.collector.PoliteHttpClient'), patch.object(EorzeaCollectionCollector, 'download_images', return_value=[]) as download, patch.object(downloader, 'current_best_image_coverage', return_value={'ok':True}):
                self.assertEqual(downloader.main(), 0)
                self.assertEqual([row.source_image_url for row in download.call_args.args[0]], [BASE+part+'.png' for part in expected])

    def test_skipped_previews_are_discovered_but_not_reported_as_missing_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); metadata = root / META_PATH; metadata.mkdir(parents=True)
            rows = [{'source_image_url': BASE+part+'.png', 'accessible': True} for part in ('front', 'back', 'body')]
            (metadata/'image_url_manifest_raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            default = image_snapshot(root)
            self.assertEqual(len(default['discovered']), 3)
            self.assertEqual(default['download_required'], [BASE+'body.png'])
            self.assertEqual(len(image_snapshot(root, include_full_previews=True)['download_required']), 3)
            default['downloaded'] = [BASE+'body.png']
            result = update_result({}, default, candidate={}, download={}, warnings=0, failed=False, planned=False)
            self.assertEqual(result['missing_download_images'], 0)
            self.assertNotIn('待处理', result['message'])

    def test_lifecycle_exposes_preview_opt_in_without_changing_default_web_updates(self):
        for include in (False, True):
            args = lifecycle.parse_args(['--include-full-previews'] if include else [])
            stage = next(s for s in lifecycle.sample_stages(args, 'preview-test') if s.name == 'download-images')
            self.assertEqual('--include-full-previews' in stage.command, include)

if __name__ == '__main__':
    unittest.main()
