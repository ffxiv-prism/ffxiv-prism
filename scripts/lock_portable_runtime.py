"""Download official wheels selected by pip's dry-run report and freeze hashes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def fetch(info):
    url = info["download_info"]["url"].replace("https://download-r2.pytorch.org/", "https://download.pytorch.org/")
    filename = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])
    digest = info["download_info"]["archive_info"].get("hashes", {}).get("sha256")
    if not digest:
        api = f"https://pypi.org/pypi/{info['metadata']['name']}/{info['metadata']['version']}/json"
        with urllib.request.urlopen(api, timeout=30) as response:
            package = json.load(response)
        file = next(r for r in package["urls"] if r["filename"] == filename)
        digest, url = file["digests"]["sha256"], file["url"]
    target = ROOT / "work/wheels" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        print(f"Downloading {filename}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as source, target.with_suffix(".part").open("wb") as output:
            while data := source.read(1024 * 1024):
                output.write(data)
        target.with_suffix(".part").replace(target)
    with target.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"Hash mismatch: {filename}")
    return {"filename": filename, "size": target.stat().st_size, "sha256": digest, "urls": [url]}

def main():
    report = json.loads((ROOT / "work/runtime-resolve.json").read_text("utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        wheels = sorted(pool.map(fetch, report["install"]), key=lambda x: x["filename"])
    identity = hashlib.sha256(json.dumps(wheels, sort_keys=True).encode()).hexdigest()[:16]
    (ROOT / "manifests").mkdir(exist_ok=True)
    (ROOT / "manifests/runtime.json").write_text(json.dumps({"schema": 1, "id": "cpu-" + identity, "wheels": wheels}, indent=2), "utf-8")
    url = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
    target = ROOT / "work/python-3.12.10-embed-amd64.zip"
    if not target.is_file():
        with urllib.request.urlopen(url, timeout=60) as source:
            target.write_bytes(source.read())
    print(f"Runtime locked: {len(wheels)} wheels, {sum(r['size'] for r in wheels)/1024**2:.1f} MiB", flush=True)

if __name__ == "__main__":
    main()
