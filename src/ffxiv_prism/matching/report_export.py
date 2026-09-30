"""Render the original HTML report with a local headless Chromium browser."""
from __future__ import annotations

import base64
import math
import logging
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ffxiv_prism.collectors.eorzea_collection.visible_browser import (
    CdpChrome,
    close_visible_driver,
    find_chrome_executable,
    reserve_local_port,
    wait_for_debug_endpoint,
)
from ffxiv_prism.paths import portable_path, HOME_ROOT


def find_report_browser() -> Path:
    try:
        return find_chrome_executable()
    except FileNotFoundError:
        candidates = [shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome", "msedge")]
        candidates += [
            str(Path(os.environ.get(key, default)) / "Microsoft/Edge/Application/msedge.exe")
            for key, default in (("PROGRAMFILES", "C:/Program Files"), ("PROGRAMFILES(X86)", "C:/Program Files (x86)"), ("LOCALAPPDATA", ""))
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return Path(candidate)
    raise FileNotFoundError("导出报告 PNG 需要本机安装 Chrome、Edge 或 Chromium。")


def render_report_png(html_path: Path, *, width: int = 1280) -> Path:
    """Capture the entire report at a fixed desktop width, using its own CSS."""
    html_path = html_path.resolve(strict=True)
    if width < 320:
        raise ValueError("Report viewport must be at least 320 pixels wide")
    destination = html_path.with_suffix(".png")
    browser = find_report_browser()
    temporary_root = HOME_ROOT / "work/temp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=temporary_root, prefix="prism-report-", ignore_cleanup_errors=True) as profile:
        port = reserve_local_port()
        command = [str(browser), "--headless=new", "--hide-scrollbars", "--no-first-run",
                   "--no-default-browser-check", "--disable-background-networking", "--disable-extensions",
                   "--remote-debugging-address=127.0.0.1", f"--remote-debugging-port={port}",
                   f"--user-data-dir={profile}", "about:blank"]
        options: dict[str, Any] = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
        driver = None
        try:
            wait_for_debug_endpoint(port, process)
            driver = CdpChrome.connect(port, process, preferred_url="about:blank")
            driver.command("Emulation.setDeviceMetricsOverride", {
                "width": width, "height": 900, "deviceScaleFactor": 1, "mobile": False,
            })
            driver.get(html_path.as_uri())
            ready = driver.evaluate("""Promise.race([
              (async () => {
                await document.fonts.ready;
                await Promise.all([...document.images].map(img => img.complete ? Promise.resolve() :
                  new Promise(resolve => { img.addEventListener('load', resolve, {once:true});
                    img.addEventListener('error', resolve, {once:true}); })));
                await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                return true;
              })(),
              new Promise(resolve => setTimeout(() => resolve(false), 30000))
            ])""", await_promise=True)
            if not ready:
                raise TimeoutError("报告字体或图片加载超时，未生成完整 PNG。")
            size = driver.command("Page.getLayoutMetrics")["cssContentSize"]
            screenshot = driver.command("Page.captureScreenshot", {
                "format": "png", "captureBeyondViewport": True,
                "clip": {"x": 0, "y": 0, "width": math.ceil(size["width"]),
                         "height": math.ceil(size["height"]), "scale": 1},
            })
            content = base64.b64decode(screenshot["data"], validate=True)
            if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("浏览器未返回有效 PNG。")
            temporary = destination.with_suffix(".png.tmp")
            try:
                temporary.write_bytes(content)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            if driver is not None:
                close_visible_driver(driver)
            elif process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    return destination


def write_report_bundle(output_dir: Path, payload: dict[str, Any]) -> Path | None:
    """Keep HTML/JSON available with an explicit warning if PNG export fails."""
    from ffxiv_prism.matching.inference import write_html_report, write_json_report

    html_path = output_dir / "report.html"
    png_path = output_dir / "report.png"
    artifacts = payload.setdefault("artifacts", {})
    artifacts.pop("report_png", None)
    # A failed rerun must not leave a previous result masquerading as this PNG.
    png_path.unlink(missing_ok=True)
    write_html_report(html_path, payload)
    print("stage=report-png", flush=True)
    try:
        render_report_png(html_path)
    except Exception as exc:
        warning = f"报告 PNG 导出失败，HTML/JSON 已保留：{exc}"
        logging.getLogger(__name__).warning("%s（%s）", warning, output_dir)
        png_path = None
    else:
        artifacts["report_png"] = portable_path(png_path)
    write_json_report(output_dir / "results.json", payload)
    return png_path
