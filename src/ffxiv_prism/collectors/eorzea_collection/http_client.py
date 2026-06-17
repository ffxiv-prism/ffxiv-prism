# 功能：实现带缓存、限速和基础 robots 检查的 HTTP 下载客户端。

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ffxiv_prism.collectors.eorzea_collection.url_utils import extension_from_url


@dataclass(frozen=True)
class FetchResult:
    url: str
    local_path: Path
    from_cache: bool
    status: int | None
    content_type: str | None


class PoliteHttpClient:
    """Small stdlib HTTP client designed for slow, cache-first collection."""

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str = "ffxiv-prism-local-dataset/0.1",
        delay_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
        respect_robots: bool = True,
        require_robots: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.respect_robots = respect_robots
        self.require_robots = require_robots
        self._last_request_at = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_text(self, url: str, target_path: Path | None = None) -> tuple[str, FetchResult]:
        target = target_path or self._cache_path(url, ".html")
        result = self.fetch_bytes(url, target)
        return result.local_path.read_text(encoding="utf-8", errors="replace"), result

    def fetch_bytes(self, url: str, target_path: Path | None = None) -> FetchResult:
        target = target_path or self._cache_path(url, extension_from_url(url))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            return FetchResult(url=url, local_path=target, from_cache=True, status=None, content_type=None)

        self._check_allowed(url)
        self._wait()
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
                target.write_bytes(body)
                return FetchResult(
                    url=url,
                    local_path=target,
                    from_cache=False,
                    status=getattr(response, "status", None),
                    content_type=response.headers.get("Content-Type"),
                )
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} while fetching {url}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error while fetching {url}: {exc.reason}") from exc

    def _cache_path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}{suffix}"

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _check_allowed(self, url: str) -> None:
        if not self.respect_robots:
            return
        parsed = urlparse(url)
        robots_key = f"{parsed.scheme}://{parsed.netloc}"
        robot = self._robots.get(robots_key)
        if robots_key not in self._robots:
            robot = self._load_robots(robots_key)
            self._robots[robots_key] = robot
        if robot is None:
            if self.require_robots:
                raise RuntimeError(f"robots.txt unavailable for {robots_key}")
            return
        if not robot.can_fetch(self.user_agent, url):
            raise RuntimeError(f"robots.txt disallows fetching {url}")

    def _load_robots(self, site_root: str) -> urllib.robotparser.RobotFileParser | None:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"{site_root}/robots.txt")
        try:
            parser.read()
        except Exception:
            return None
        return parser
