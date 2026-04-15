#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper_figures"
WINDOW_SIZE = "1600,900"


def resolve_browser() -> str:
    for candidate in [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ]:
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("No supported headless browser found for PNG rendering.")


def render_svg_to_png(browser: str, svg_path: Path) -> None:
    png_path = svg_path.with_suffix(".png")
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={WINDOW_SIZE}",
        f"--screenshot={png_path}",
        svg_path.as_uri(),
    ]
    _ = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def main() -> None:
    browser = resolve_browser()
    for svg_path in sorted(FIGURE_DIR.glob("Figure_*.svg")):
        render_svg_to_png(browser, svg_path)


if __name__ == "__main__":
    main()
