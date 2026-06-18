from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bs4 import BeautifulSoup, Tag

from ffxiv_prism.jsonl import append_jsonl, read_jsonl, write_json
from ffxiv_prism.paths import CACHE_DIR, EORZEA_METADATA_DIR


DEFAULT_ITEM_NAME = "Augmented Courtly Lover's Surcoat of Maiming"
DEFAULT_SEARCH_URL = "https://ff14.huijiwiki.com/wiki/ItemSearch"
DEFAULT_WARMUP_URL = "https://ff14.huijiwiki.com/"
DEFAULT_BATCH_INPUT = EORZEA_METADATA_DIR / "gear_pieces_raw.jsonl"
DEFAULT_BATCH_CACHE = CACHE_DIR / "huiji_item_zh_name_cache.jsonl"
DEFAULT_BATCH_SUMMARY = CACHE_DIR / "huiji_item_zh_name_batch_summary.json"
DEFAULT_USER_DATA_DIR = CACHE_DIR / "huiji_chrome_profile"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class ItemNameMatch:
    name_en: str
    name_zh: str
    item_url: str
    search_url: str


@dataclass(frozen=True)
class SearchPlan:
    original_name: str
    query_name: str
    gender: str | None = None


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_missing:
        return run_batch(args)

    with HuijiSeleniumClient(
        search_url=args.search_url,
        warmup_url=args.warmup_url,
        user_agent=args.user_agent,
        proxy=args.proxy,
        cookie_header=args.cookie,
        cookie_file=args.cookie_file,
        user_data_dir=args.user_data_dir,
        browser_wait=args.browser_wait,
        content_wait=args.content_wait,
        headless=args.headless,
    ) as client:
        match = client.find(args.name)
    print_match(match, as_json=args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search HuijiWiki ItemSearch by English FFXIV item name and return Chinese item names."
    )
    parser.add_argument("name", nargs="?", default=DEFAULT_ITEM_NAME, help="English item name to search.")
    parser.add_argument("--search-url", default=DEFAULT_SEARCH_URL, help="HuijiWiki ItemSearch URL.")
    parser.add_argument("--warmup-url", default=DEFAULT_WARMUP_URL, help="URL to open before searching.")
    parser.add_argument(
        "--cookie",
        default=os.environ.get("HUIJIWIKI_COOKIE"),
        help="Browser Cookie header. Defaults to HUIJIWIKI_COOKIE.",
    )
    parser.add_argument(
        "--cookie-file",
        type=Path,
        default=None,
        help="Cookie file in Netscape cookies.txt, JSON dict, or JSON browser-export list format.",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("HUIJIWIKI_USER_AGENT", DEFAULT_USER_AGENT),
        help="Browser User-Agent header. Defaults to HUIJIWIKI_USER_AGENT.",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Optional HTTP(S) proxy URL, for example http://127.0.0.1:10808. Defaults to Chrome/system networking.",
    )
    parser.add_argument(
        "--user-data-dir",
        type=Path,
        default=DEFAULT_USER_DATA_DIR,
        help="Persistent Chrome profile directory. Reusing it reduces repeated verification.",
    )
    parser.add_argument("--browser-wait", type=float, default=120.0, help="Seconds to wait for Cloudflare to pass.")
    parser.add_argument("--content-wait", type=float, default=8.0, help="Seconds to wait for search result content.")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless. Visible mode is better for challenges.")
    parser.add_argument("--json", action="store_true", help="Print JSON for single-item mode.")

    parser.add_argument("--batch-missing", action="store_true", help="Fill missing name_zh fields in a JSONL dataset.")
    parser.add_argument("--batch-input", type=Path, default=DEFAULT_BATCH_INPUT, help="Input gear pieces JSONL.")
    parser.add_argument(
        "--batch-output",
        type=Path,
        default=None,
        help="Output JSONL. Defaults to updating --batch-input in place with a backup.",
    )
    parser.add_argument("--batch-cache", type=Path, default=DEFAULT_BATCH_CACHE, help="Incremental lookup cache JSONL.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_BATCH_SUMMARY, help="Batch summary JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit unique missing names for testing.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry names cached as failed.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch on the first failed lookup.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a backup when updating in place.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between batch lookups.")
    return parser


class HuijiSeleniumClient:
    def __init__(
        self,
        search_url: str,
        warmup_url: str,
        user_agent: str,
        proxy: str | None,
        cookie_header: str | None,
        cookie_file: Path | None,
        user_data_dir: Path,
        browser_wait: float,
        content_wait: float,
        headless: bool,
    ) -> None:
        self.search_url = search_url
        self.warmup_url = warmup_url
        self.user_agent = user_agent
        self.proxy = proxy
        self.cookie_header = cookie_header
        self.cookie_file = cookie_file
        self.user_data_dir = user_data_dir
        self.browser_wait = browser_wait
        self.content_wait = content_wait
        self.headless = headless
        self.driver: Any | None = None

    def __enter__(self) -> "HuijiSeleniumClient":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def start(self) -> None:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except ImportError as exc:
            raise SystemExit("Missing dependency: selenium. Install project dependencies first.") from exc

        self.user_data_dir = self.user_data_dir.resolve()
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        options = Options()
        options.add_argument(f"--user-agent={self.user_agent}")
        options.add_argument(f"--user-data-dir={self.user_data_dir}")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--lang=zh-CN")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        if self.headless:
            options.add_argument("--headless=new")
        if self.proxy:
            options.add_argument(f"--proxy-server={self.proxy}")

        self.driver = webdriver.Chrome(options=options)
        apply_selenium_stealth(self.driver)
        self.seed_cookies()

    def close(self) -> None:
        if self.driver is not None:
            self.driver.quit()
            self.driver = None

    def seed_cookies(self) -> None:
        assert self.driver is not None
        cookies = build_selenium_cookies(self.cookie_header, self.cookie_file)
        if not cookies:
            return
        self.driver.get(self.warmup_url)
        for cookie in cookies:
            try:
                self.driver.add_cookie(cookie)
            except Exception:
                pass

    def find(self, item_name_en: str) -> ItemNameMatch:
        assert self.driver is not None
        plan = build_search_plan(item_name_en)
        target_url = with_query(self.search_url, {"name": plan.query_name})
        item_url = None
        for attempt in range(2):
            self.driver.get(target_url)
            if not self.headless:
                print(
                    "Chrome automation window opened. Complete HuijiWiki/Cloudflare verification there if prompted.",
                    file=sys.stderr,
                    flush=True,
            )
            wait_until_not_challenge(self.driver, self.browser_wait)
            wait_until_search_has_result_or_settled(self.driver, target_url, plan, self.content_wait)
            item_url = extract_item_url(self.driver.page_source, target_url, plan)
            if item_url is not None:
                break
            if attempt == 0:
                time.sleep(1.0)
        if item_url is None:
            raise RuntimeError(f"No HuijiWiki item link found in search results for: {item_name_en!r}")

        name_zh = extract_title_from_item_url(item_url)
        if not name_zh:
            self.driver.get(item_url)
            wait_until_not_challenge(self.driver, self.browser_wait)
            name_zh = extract_page_title(self.driver.page_source)
        if not name_zh:
            raise RuntimeError(f"No page title found for item URL: {item_url}")
        return ItemNameMatch(name_en=item_name_en, name_zh=name_zh, item_url=item_url, search_url=target_url)


def run_batch(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.batch_input)
    names = collect_missing_names(rows)
    if args.limit is not None:
        names = names[: args.limit]

    cache = load_lookup_cache(args.batch_cache)
    pending = [
        name
        for name in names
        if name not in cache or (args.retry_failed and cache[name].get("status") == "failed")
    ]

    started_at = time.monotonic()
    stats = {"matched": 0, "failed": 0, "cached": len(names) - len(pending)}
    with HuijiSeleniumClient(
        search_url=args.search_url,
        warmup_url=args.warmup_url,
        user_agent=args.user_agent,
        proxy=args.proxy,
        cookie_header=args.cookie,
        cookie_file=args.cookie_file,
        user_data_dir=args.user_data_dir,
        browser_wait=args.browser_wait,
        content_wait=args.content_wait,
        headless=args.headless,
    ) as client:
        for index, name in enumerate(pending, start=1):
            try:
                match = client.find(name)
                record = {
                    "name_en": match.name_en,
                    "name_zh": match.name_zh,
                    "item_url": match.item_url,
                    "search_url": match.search_url,
                    "status": "matched",
                    "finished_at": timestamp(),
                }
                stats["matched"] += 1
            except Exception as exc:
                record = {
                    "name_en": name,
                    "name_zh": None,
                    "item_url": None,
                    "search_url": with_query(args.search_url, {"name": name}),
                    "status": "failed",
                    "error": str(exc),
                    "finished_at": timestamp(),
                }
                stats["failed"] += 1
                append_jsonl(args.batch_cache, [record])
                print(f"[{index}/{len(pending)}] failed: {name}: {exc}", file=sys.stderr, flush=True)
                cache[name] = record
                if args.stop_on_error:
                    raise
                continue

            cache[name] = record
            append_jsonl(args.batch_cache, [record])
            print(
                f"[{index}/{len(pending)}] matched: {name} -> {record['name_zh']}",
                file=sys.stderr,
                flush=True,
            )
            if args.delay > 0:
                time.sleep(args.delay)

    output = args.batch_output or args.batch_input
    enriched_rows, row_stats = enrich_rows(rows, cache)
    write_jsonl_with_optional_backup(args.batch_input, output, enriched_rows, args.no_backup)
    summary = {
        "input": str(args.batch_input),
        "output": str(output),
        "cache": str(args.batch_cache),
        "unique_missing_names": len(names),
        "unique_pending_before_run": len(pending),
        "cached_before_run": stats["cached"],
        "matched_this_run": stats["matched"],
        "failed_this_run": stats["failed"],
        "rows_filled": row_stats["rows_filled"],
        "rows_still_missing": row_stats["rows_still_missing"],
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
        "finished_at": timestamp(),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def wait_until_not_challenge(driver: Any, timeout: float) -> None:
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(driver, timeout, poll_frequency=0.5).until(
        lambda current: not page_has_cloudflare_challenge(current.page_source)
    )


def wait_until_search_has_result_or_settled(
    driver: Any,
    base_url: str,
    plan: SearchPlan,
    timeout: float,
) -> None:
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.support.ui import WebDriverWait

    def ready(current: Any) -> bool:
        if page_has_cloudflare_challenge(current.page_source):
            return False
        html = current.page_source
        if extract_item_url(html, base_url, plan) is not None:
            return True
        return page_has_settled_empty_itemsearch(html)

    try:
        WebDriverWait(driver, timeout, poll_frequency=0.25).until(ready)
    except TimeoutException:
        pass


def page_has_settled_empty_itemsearch(html: str) -> bool:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    if "\u5171\u6709 0 \u4e2a\u7b26\u5408\u6761\u4ef6\u7684\u7269\u54c1" in text:
        return True
    if "\u6ca1\u6709\u627e\u5230" in text and "\u7269\u54c1" in text:
        return True
    return False


def apply_selenium_stealth(driver: Any) -> None:
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = window.chrome || { runtime: {} };
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
  parameters.name === 'notifications'
    ? Promise.resolve({ state: Notification.permission })
    : originalQuery(parameters)
);
""",
            },
        )
    except Exception:
        pass


def build_selenium_cookies(cookie_header: str | None, cookie_file: Path | None) -> list[dict[str, Any]]:
    cookies: list[dict[str, Any]] = []
    if cookie_header:
        for name, value in parse_cookie_header(cookie_header).items():
            cookies.append({"name": name, "value": value, "domain": ".huijiwiki.com", "path": "/"})
    if cookie_file is None:
        return cookies

    text = cookie_file.read_text(encoding="utf-8").strip()
    if not text:
        return cookies
    if text[0] in "[{":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for name, value in payload.items():
                cookies.append({"name": str(name), "value": str(value), "domain": ".huijiwiki.com", "path": "/"})
        elif isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict) and row.get("name") is not None and row.get("value") is not None:
                    cookies.append(
                        {
                            "name": str(row["name"]),
                            "value": str(row["value"]),
                            "domain": str(row.get("domain") or ".huijiwiki.com"),
                            "path": str(row.get("path") or "/"),
                        }
                    )
    else:
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) == 7:
                domain, _include_subdomains, path, _secure, _expires, name, value = fields
                cookies.append({"name": name, "value": value, "domain": domain, "path": path})
    return cookies


def page_has_cloudflare_challenge(html: str) -> bool:
    prefix = html[:10000]
    text = prefix.casefold()
    return (
        ("just a moment" in text and "cloudflare" in text)
        or ("cf-turnstile-response" in text)
        or ("challenge-platform" in text)
        or ("\u8bf7\u7a0d\u5019" in prefix and "cloudflare" in text)
        or ("\u6b63\u5728\u8fdb\u884c\u5b89\u5168\u9a8c\u8bc1" in prefix)
        or ("\u9a8c\u8bc1\u6210\u529f" in prefix and "cloudflare" in text)
    )


def with_query(url: str, params: dict[str, str]) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(params)
    encoded_query = urlencode(query, quote_via=quote, safe="'")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, encoded_query, parts.fragment))


def extract_item_url(html: str, base_url: str, plan: SearchPlan | str) -> str | None:
    if isinstance(plan, str):
        plan = build_search_plan(plan)
    soup = BeautifulSoup(html, "html.parser")
    normalized_target = normalize(plan.query_name)
    exact_text_node = soup.find(string=lambda value: value is not None and normalized_target in normalize(value))
    if exact_text_node is not None:
        candidate = nearest_item_link(exact_text_node.parent, base_url, plan)
        if candidate is not None:
            return candidate

    candidates: list[tuple[int, str]] = []
    for link in soup.find_all("a", href=True):
        text = normalize(link.get_text(" ", strip=True))
        title = normalize(str(link.get("title") or ""))
        href = str(link.get("href") or "")
        if normalized_target in text or normalized_target in title or normalized_target in normalize(unquote(href)):
            item_url = normalize_item_url(href, base_url)
            if item_url is not None:
                candidates.append((score_candidate_link(link, item_url, plan), item_url))
    if candidates:
        candidates.sort(key=lambda row: row[0], reverse=True)
        return candidates[0][1]
    return extract_first_itemsearch_result_url(soup, base_url, plan)


def extract_first_itemsearch_result_url(soup: BeautifulSoup, base_url: str, plan: SearchPlan) -> str | None:
    scored: list[tuple[int, str]] = []
    for link in soup.find_all("a", href=True):
        href = str(link.get("href") or "")
        item_url = normalize_item_url(href, base_url)
        if item_url is None:
            continue
        text = link.get_text(" ", strip=True)
        if not text or len(text) > 80:
            continue
        score = score_itemsearch_link(link) + score_candidate_link(link, item_url, plan)
        if score > 0:
            scored.append((score, item_url))
    if not scored:
        return None
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored[0][1]


def score_itemsearch_link(link: Tag) -> int:
    score = 1
    current: Tag | None = link
    for depth in range(4):
        if current is None:
            break
        text = current.get_text(" ", strip=True)
        if "Lv" in text or "\u7b49\u7ea7" in text or "\u54c1\u7ea7" in text:
            score += 10 - depth
        if current.find("img"):
            score += 2
        current = current.parent if isinstance(current.parent, Tag) else None
    return score


def nearest_item_link(node: Tag | None, base_url: str, plan: SearchPlan) -> str | None:
    current = node
    candidates: list[tuple[int, str]] = []
    for _depth in range(6):
        if current is None:
            break
        if current.name == "a" and current.get("href"):
            item_url = normalize_item_url(str(current["href"]), base_url)
            if item_url is not None:
                candidates.append((score_candidate_link(current, item_url, plan), item_url))
        link = current.find("a", href=True)
        if link is not None:
            item_url = normalize_item_url(str(link["href"]), base_url)
            if item_url is not None:
                candidates.append((score_candidate_link(link, item_url, plan), item_url))
        current = current.parent if isinstance(current.parent, Tag) else None
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[0], reverse=True)
    return candidates[0][1]


def build_search_plan(item_name_en: str) -> SearchPlan:
    name = item_name_en.strip()
    for suffix, gender in ((" [F]", "female"), (" [M]", "male")):
        if name.endswith(suffix):
            return SearchPlan(original_name=item_name_en, query_name=name[: -len(suffix)].strip(), gender=gender)
    return SearchPlan(original_name=item_name_en, query_name=name, gender=None)


def score_candidate_link(link: Tag, item_url: str, plan: SearchPlan) -> int:
    score = 0
    title_text = unquote(urlsplit(item_url).path.rsplit("/wiki/", 1)[-1])
    text_parts = [title_text, link.get_text(" ", strip=True), str(link.get("title") or "")]
    parent = link.parent if isinstance(link.parent, Tag) else None
    if parent is not None:
        text_parts.append(parent.get_text(" ", strip=True))
    haystack = normalize(" ".join(text_parts))
    query_tokens = [token for token in normalize(plan.query_name).split() if token]
    score += sum(2 for token in query_tokens if token in haystack)
    if normalize(plan.query_name) in haystack:
        score += 20
    if plan.gender:
        gender_score = score_gender_match(haystack, plan.gender)
        if gender_score < 0:
            return -1000
        score += gender_score
    return score


def score_gender_match(text: str, gender: str) -> int:
    female_markers = ("[f]", "(f)", " f ", "female", "women", "woman", "\u5973", "\u5973\u5f0f", "\u5973\u6027", "\u5973\u58eb")
    male_markers = ("[m]", "(m)", " m ", "male", "men", "man", "\u7537", "\u7537\u5f0f", "\u7537\u6027", "\u7537\u58eb")
    wanted = female_markers if gender == "female" else male_markers
    unwanted = male_markers if gender == "female" else female_markers
    wanted_hit = any(marker in text for marker in wanted)
    unwanted_hit = any(marker in text for marker in unwanted)
    if wanted_hit:
        return 100
    if unwanted_hit:
        return -100
    return 0


def normalize_item_url(href: str, base_url: str) -> str | None:
    url = urljoin(base_url, href)
    parts = urlsplit(url)
    if parts.netloc and "ff14.huijiwiki.com" not in parts.netloc:
        return None
    if "/wiki/" not in parts.path:
        return None
    title = parts.path.rsplit("/wiki/", 1)[-1]
    if not title or title == "ItemSearch":
        return None
    decoded_title = unquote(title)
    if not is_item_page_title(decoded_title):
        return None
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def is_item_page_title(title: str) -> bool:
    return title.startswith("Item:") or title.startswith("\u7269\u54c1:")


def extract_page_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for selector in ("h1#firstHeading", ".mw-page-title-main", "meta[property='og:title']", "title"):
        node = soup.select_one(selector)
        if node is None:
            continue
        if node.name == "meta":
            title = str(node.get("content") or "").strip()
        else:
            title = node.get_text(" ", strip=True)
        title = clean_title(title)
        if title and not page_title_is_challenge(title):
            return title
    return None


def extract_title_from_item_url(item_url: str) -> str | None:
    title = urlsplit(item_url).path.rsplit("/wiki/", 1)[-1]
    if not title:
        return None
    title = unquote(title)
    if ":" in title:
        title = title.split(":", 1)[1]
    return clean_title(title)


def clean_title(title: str) -> str:
    if " - " in title:
        title = title.split(" - ", 1)[0]
    title = " ".join(title.strip().split())
    return strip_known_namespace(title)


def strip_known_namespace(title: str) -> str:
    for namespace in ("Item", "\u7269\u54c1"):
        prefix = f"{namespace}:"
        if title.startswith(prefix):
            return title[len(prefix) :].strip()
    return title


def page_title_is_challenge(title: str) -> bool:
    return title.casefold() in {"just a moment...", "please wait..."} or title == "\u8bf7\u7a0d\u5019\u2026"


def parse_cookie_header(cookie_header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in cookie_header.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name:
            cookies[name] = value
    return cookies


def collect_missing_names(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("name_zh"):
            continue
        name = str(row.get("name_en") or row.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def load_lookup_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        name = str(row.get("name_en") or "").strip()
        if name:
            cache[name] = row
    return cache


def enrich_rows(rows: list[dict[str, Any]], cache: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    enriched_rows: list[dict[str, Any]] = []
    rows_filled = 0
    rows_still_missing = 0
    for row in rows:
        enriched = dict(row)
        name = str(enriched.get("name_en") or enriched.get("name") or "").strip()
        cached = cache.get(name)
        if not enriched.get("name_zh") and cached and cached.get("status") == "matched" and cached.get("name_zh"):
            enriched["name_en"] = name
            enriched["name_zh"] = cached["name_zh"]
            enriched["name_zh_source"] = cached.get("item_url")
            enriched["name_zh_match_status"] = "matched"
            enriched["name_zh_candidates"] = [cached["name_zh"]]
            rows_filled += 1
        if not enriched.get("name_zh"):
            rows_still_missing += 1
        enriched_rows.append(enriched)
    return enriched_rows, {"rows_filled": rows_filled, "rows_still_missing": rows_still_missing}


def write_jsonl_with_optional_backup(input_path: Path, output_path: Path, rows: Iterable[dict[str, Any]], no_backup: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == input_path.resolve() and input_path.exists() and not no_backup:
        backup = input_path.with_suffix(input_path.suffix + ".before_huiji_zh_names")
        if not backup.exists():
            shutil.copy2(input_path, backup)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_output.write_text("", encoding="utf-8")
    append_jsonl(tmp_output, rows)
    tmp_output.replace(output_path)


def print_match(match: ItemNameMatch, as_json: bool) -> None:
    payload = {
        "name_en": match.name_en,
        "name_zh": match.name_zh,
        "item_url": match.item_url,
        "search_url": match.search_url,
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(match.name_zh)


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
