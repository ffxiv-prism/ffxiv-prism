"""Exercise report image URLs with the actual portable directory configuration."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PortableReportImageTests(unittest.TestCase):
    def test_report_images_load_without_exposing_other_installation_files(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PRISM_HOME=directory, PRISM_PORTABLE='1',
                       PRISM_RESOURCE_ROOT=str(Path(directory) / 'resources/gallery-test'),
                       PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / 'src'))))
            result = subprocess.run([sys.executable, '-B', '-c', r'''
import html
import http.client
from pathlib import Path
import re
import threading
from urllib.parse import quote
from scripts import web_app as web

output = web.OUTPUT_ROOT / 'report-test'
output.mkdir(parents=True)
gallery = web.PROJECT_ROOT / 'data/images'
gallery.mkdir(parents=True)
assert web.DATA_IMAGE_ROOT == gallery
images = [output / 'source.png', output / 'overlay.jpg', output / 'crops.jpg',
          gallery / 'equipment preview.png']
for image in images:
    image.write_bytes(b'image fixture')
(output / 'report.html').write_text(''.join(
    '<img src="' + image.as_uri() + '">' for image in images), encoding='utf-8')
secret = web.HOME_ROOT / 'private.txt'
secret.write_text('must not be served')
server = web.PrismServer(('127.0.0.1', 0))
job = server.registry.reserve(kind='recognition', title='test', output_dir=output, device='cpu')
job.status = 'succeeded'
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
def fetch(url):
    conn = http.client.HTTPConnection(*server.server_address, timeout=10)
    try:
        conn.request('GET', url)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()
try:
    status, report = fetch('/api/jobs/' + job.id + '/report')
    assert status == 200
    urls = re.findall(r'src="([^"]+)"', report.decode('utf-8'))
    assert len(urls) == len(images)
    for url in urls:
        assert fetch(html.unescape(url)) == (200, b'image fixture'), url
    for path in (secret, gallery / '../private.txt'):
        assert fetch('/api/file?path=' + quote(str(path), safe=''))[0] == 403
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
'''], env=env, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
