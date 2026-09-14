#!/usr/bin/env python3
"""Start the app."""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
WEB = ROOT / "web"
HOST, PORT = "127.0.0.1", 8000


def run(command: list[str], cwd: Path) -> None:
    printable = " ".join(command)
    print(f"  {printable}")
    if subprocess.run(command, cwd=cwd).returncode != 0:
        raise SystemExit(f"failed: {printable}")


def npm() -> str:
    # windows sometimes uses npm.cmd
    found = shutil.which("npm") or shutil.which("npm.cmd")
    if not found:
        raise SystemExit(
            "npm was not found. Install Node.js (https://nodejs.org) and try again."
        )
    return found


def build_interface(force: bool = False) -> None:
    if (WEB / "dist" / "index.html").exists() and not force:
        return
    print("building the interface (first run only, takes a minute)")
    if not (WEB / "node_modules").is_dir():
        run([npm(), "install"], cwd=WEB)
    run([npm(), "run", "build"], cwd=WEB)


def migrate() -> None:
    print("preparing the database")
    run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=SERVER)


def open_when_ready(url: str) -> None:
    # wait until the api is up
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=1)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="hot-reloading frontend")
    parser.add_argument("--no-open", action="store_true", help="don't open a browser")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the interface")
    args = parser.parse_args()

    sys.path.insert(0, str(SERVER))
    migrate()

    from forge.config import get_settings

    settings = get_settings()
    print(f"data lives in {settings.data_dir}")

    if args.dev:
        build_needed = not (WEB / "node_modules").is_dir()
        if build_needed:
            run([npm(), "install"], cwd=WEB)
        vite = subprocess.Popen([npm(), "run", "dev"], cwd=WEB)
        url = "http://localhost:5173"
    else:
        build_interface(force=args.rebuild)
        vite = None
        url = f"http://{HOST}:{PORT}"

    if not args.no_open:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    print(f"\nForge is running at {url}   (ctrl-c to stop)\n")
    try:
        os.chdir(SERVER)
        import uvicorn

        uvicorn.run("forge.main:app", host=HOST, port=PORT, log_level="warning")
    finally:
        if vite is not None:
            vite.terminate()


if __name__ == "__main__":
    main()
