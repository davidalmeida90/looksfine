"""Headless screenshots of LooksFine pages, for a README or a check after a change. No browser window opens.
Needs Playwright (pip install playwright, then: playwright install chromium) and the server running.
    python shots.py OUTDIR name=#/route [name=#/route ...] [--port 8765] [--width 1440] [--light]
    python shots.py shots exp=#/ jobs=#/jobs final=#/final llm=#/llm
Writes OUTDIR/<name>.png (full page) and prints any console errors the page logged.
In Git Bash, prefix with MSYS_NO_PATHCONV=1 so '#/route' is not turned into a Windows path.
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("pages", nargs="+")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--width", type=int, default=1440)
ap.add_argument("--light", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={"width": a.width, "height": 900}, device_scale_factor=1)
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    if a.light:
        pg.add_init_script("try { localStorage.setItem('lf-theme', 'light') } catch (e) {}")
    for spec in a.pages:
        name, route = spec.split("=", 1)
        pg.goto(f"http://127.0.0.1:{a.port}/{route}")
        pg.wait_for_load_state("networkidle")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=str(out / f"{name}.png"), full_page=True)
        print(f"{name}: {out / (name + '.png')}")
    b.close()
print("console errors: " + ("; ".join(errs) if errs else "none"))
