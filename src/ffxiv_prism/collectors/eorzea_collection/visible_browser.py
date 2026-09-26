from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

import websocket

from ffxiv_prism.paths import CACHE_DIR


DEFAULT_PROFILE_DIR = CACHE_DIR / "eorzea_collection_browser"


def create_visible_driver(
    profile_dir: Path,
    *,
    manual_handoff_url: str | None = None,
) -> "CdpChrome":
    """Start normal Chrome and control it without enabling WebDriver mode."""

    profile_dir.mkdir(parents=True, exist_ok=True)
    return create_cdp_driver(
        profile_dir,
        manual_handoff_url or "about:blank",
        wait_for_handoff=bool(manual_handoff_url),
    )


def create_cdp_driver(
    profile_dir: Path,
    url: str,
    *,
    wait_for_handoff: bool,
) -> "CdpChrome":
    """Launch normal Chrome and attach through CDP without WebDriver."""

    chrome_path = find_chrome_executable()
    debug_port = reserve_local_port()
    process = subprocess.Popen(
        [
            str(chrome_path),
            f"--remote-debugging-port={debug_port}",
            "--remote-allow-origins=*",
            f"--user-data-dir={profile_dir.resolve()}",
            "--new-window",
            url,
        ]
    )
    try:
        wait_for_debug_endpoint(debug_port, process)
        if wait_for_handoff:
            if debug_target_is_ready(debug_port, url, timeout_seconds=5.0):
                print(
                    "manual_verification=already_validated browser=normal_chrome",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    "manual_verification=handoff "
                    "browser=normal_chrome action=complete_verification_then_press_enter",
                    file=sys.stderr,
                    flush=True,
                )
                input("人工验证完成后按 Enter 继续流水线: ")
        if process.poll() is not None:
            raise RuntimeError("Normal Chrome closed before the pipeline could attach")
        driver = CdpChrome.connect(debug_port, process, preferred_url=url)
        return driver
    except BaseException:
        if process.poll() is None:
            process.terminate()
        raise


def close_visible_driver(driver: "CdpChrome") -> None:
    process = driver.process
    try:
        driver.quit()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def find_chrome_executable() -> Path:
    candidates = [
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        str(Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe"),
        str(Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Google/Chrome/Application/chrome.exe"),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Google Chrome executable was not found")


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def wait_for_debug_endpoint(
    port: int,
    process: subprocess.Popen[bytes],
    timeout_seconds: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/json/version"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Normal Chrome closed before its debug endpoint was ready")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if getattr(response, "status", 200) == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    raise TimeoutError("Normal Chrome debug endpoint did not become ready")


class CdpChrome:
    def __init__(
        self,
        connection: websocket.WebSocket,
        process: subprocess.Popen[bytes],
    ) -> None:
        self.connection = connection
        self.process = process
        self.next_command_id = 0
        self.command("Page.enable")
        self.command("Runtime.enable")

    @classmethod
    def connect(
        cls,
        port: int,
        process: subprocess.Popen[bytes],
        *,
        preferred_url: str,
    ) -> "CdpChrome":
        targets = read_debug_json(port, "/json/list")
        pages = [row for row in targets if row.get("type") == "page"]
        preferred = [
            row
            for row in pages
            if normalized_page_url(str(row.get("url") or ""))
            == normalized_page_url(preferred_url)
        ]
        target = (preferred or pages)[0] if pages else None
        if not target or not target.get("webSocketDebuggerUrl"):
            raise RuntimeError("Normal Chrome did not expose a controllable page")
        connection = websocket.create_connection(
            str(target["webSocketDebuggerUrl"]),
            timeout=60,
            suppress_origin=True,
        )
        return cls(connection, process)

    def command(
        self,
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.next_command_id += 1
        command_id = self.next_command_id
        self.connection.send(
            json.dumps(
                {"id": command_id, "method": method, "params": params or {}}
            )
        )
        while True:
            message = json.loads(self.connection.recv())
            if message.get("id") != command_id:
                continue
            if message.get("error"):
                raise RuntimeError(
                    f"Chrome CDP command failed: {method}: {message['error']}"
                )
            result = message.get("result")
            return result if isinstance(result, dict) else {}

    def evaluate(self, expression: str, *, await_promise: bool = False) -> object:
        result = self.command(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
        )
        remote = result.get("result")
        if not isinstance(remote, dict):
            return None
        if remote.get("subtype") == "error":
            raise RuntimeError(str(remote.get("description") or "CDP evaluation failed"))
        return remote.get("value")

    @property
    def current_url(self) -> str:
        return str(self.evaluate("location.href") or "")

    @property
    def page_source(self) -> str:
        return str(self.evaluate("document.documentElement.outerHTML") or "")

    def get(self, url: str) -> None:
        self.command("Page.navigate", {"url": url})
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                state = self.evaluate("document.readyState")
                current_url = str(self.evaluate("location.href") or "")
                if (
                    state in {"interactive", "complete"}
                    and normalized_page_url(current_url)
                    == normalized_page_url(url)
                ):
                    return
            except RuntimeError:
                pass
            time.sleep(0.1)
        try:
            current_url = str(self.evaluate("location.href") or "")
            html = str(self.evaluate("document.documentElement.outerHTML") or "")
            if (
                normalized_page_url(current_url) == normalized_page_url(url)
                and len(html) >= 1_000
            ):
                return
        except RuntimeError:
            pass
        raise TimeoutError(f"Chrome navigation did not complete: {url}")

    def quit(self) -> None:
        self.connection.close()


def read_debug_json(port: int, path: str) -> object:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=5.0
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def debug_target_is_ready(
    port: int,
    preferred_url: str,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        targets = read_debug_json(port, "/json/list")
        if isinstance(targets, list):
            for target in targets:
                if not isinstance(target, dict) or target.get("type") != "page":
                    continue
                current_url = str(target.get("url") or "")
                if normalized_page_url(current_url) != normalized_page_url(
                    preferred_url
                ):
                    continue
                title = str(target.get("title") or "").casefold()
                if "just a moment" not in title and "__cf_chl" not in current_url:
                    return True
        time.sleep(0.2)
    return False


def fetch_verified_html(
    driver: CdpChrome,
    url: str,
    *,
    wait_seconds: float,
    ready: Callable[[str], bool] | None = None,
) -> tuple[str, str]:
    """Open a page and wait for the user to finish any visible verification."""

    print(f"browser_open url={url}", file=sys.stderr, flush=True)
    if normalized_page_url(driver.current_url) != normalized_page_url(url):
        driver.get(url)
    deadline = time.monotonic() + max(wait_seconds, 1.0)
    waiting_reported = False
    while time.monotonic() < deadline:
        html = driver.page_source
        page_ready = not page_has_cloudflare_challenge(html)
        if ready is not None:
            page_ready = page_ready and ready(html)
        if page_ready:
            if waiting_reported:
                print(
                    f"manual_verification=passed url={driver.current_url}",
                    file=sys.stderr,
                    flush=True,
                )
            return html, driver.current_url
        if not waiting_reported:
            print(
                "manual_verification=waiting "
                f"timeout_seconds={wait_seconds:g} url={driver.current_url}",
                file=sys.stderr,
                flush=True,
            )
            waiting_reported = True
        time.sleep(1.0)
    raise TimeoutError(
        "Page content did not become ready. Complete the visible browser "
        "verification before the configured timeout and retry."
    )


def page_has_cloudflare_challenge(html: str) -> bool:
    text = html.casefold()
    prefix = text[:20_000]
    return (
        "cf-chl-" in prefix
        or "challenge-platform" in prefix
        or ("just a moment" in prefix and "cloudflare" in text)
        or ("performing security verification" in prefix and "cloudflare" in text)
        or ("正在进行安全验证" in prefix and "cloudflare" in text)
    )


def normalized_page_url(url: str) -> str:
    parsed = urlsplit(url)
    query = "&".join(
        part for part in parsed.query.split("&") if not part.startswith("__cf_")
    )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))
