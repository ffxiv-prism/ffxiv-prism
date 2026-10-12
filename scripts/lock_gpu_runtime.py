"""Freeze Windows CUDA wheels from official index hashes; retain CPU base dependencies."""
import hashlib
import html.parser
import json
from pathlib import Path
import urllib.parse
import urllib.request

from lock_portable_runtime import download_urls

ROOT = Path(__file__).resolve().parents[1]


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.urls.extend(v for k, v in attrs if k == 'href')


def main():
    base = json.loads((ROOT / 'manifests/runtime.json').read_text('utf-8'))
    for variant in ('cu126', 'cu130'):
        wheels = [r for r in base['wheels'] if not r['filename'].startswith(('torch-', 'torchvision-'))]
        for name, version in (('torch', '2.11.0'), ('torchvision', '0.26.0')):
            index = f'https://download.pytorch.org/whl/{variant}/{name}/'
            with urllib.request.urlopen(index, timeout=60) as response:
                links = Links()
                links.feed(response.read().decode())
            filename = f'{name}-{version}+{variant}-cp312-cp312-win_amd64.whl'
            url = next(urllib.parse.urljoin(index, u) for u in links.urls
                       if urllib.parse.unquote(urllib.parse.urlsplit(u).path).endswith('/' + filename))
            digest = urllib.parse.parse_qs(urllib.parse.urlsplit(url).fragment)['sha256'][0]
            url = url.split('#')[0].replace('https://download-r2.pytorch.org/', 'https://download.pytorch.org/')
            with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0'}), timeout=60) as response:
                size = int(response.headers['Content-Range'].split('/')[-1]) if response.status == 206 else int(response.headers['Content-Length'])
            wheels.append(dict(filename=filename, size=size, sha256=digest, urls=download_urls(url)))
        wheels.sort(key=lambda r: r['filename'])
        identity = hashlib.sha256(json.dumps(wheels, sort_keys=True).encode()).hexdigest()[:16]
        path = ROOT / f'manifests/runtime-{variant}.json'
        path.write_text(json.dumps(dict(schema=1, id=f'{variant}-{identity}', wheels=wheels), indent=2) + '\n', 'utf-8')
        print(f'{variant}: {sum(r["size"] for r in wheels)/2**30:.2f} GiB download')


if __name__ == '__main__':
    main()
