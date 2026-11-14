#!/usr/bin/env python3
"""Start the app. On a server with no screen, it prints how to open it from your laptop."""

import argparse
import getpass
import ipaddress
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
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
    if not (shutil.which("npm") or shutil.which("npm.cmd")):
        raise SystemExit(
            "The interface is not built and npm is not installed here. Build it on a computer "
            "with Node.js (python scripts/start.py --rebuild), then copy the web/dist folder "
            "to the same place on this computer."
        )
    print("building the interface (first run only, takes a minute)")
    if not (WEB / "node_modules").is_dir():
        run([npm(), "install"], cwd=WEB)
    run([npm(), "run", "build"], cwd=WEB)


def migrate() -> None:
    print("preparing the database")
    run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=SERVER)


def headless() -> bool:
    # no screen to open a browser on: a server reached over ssh
    if os.environ.get("SSH_CONNECTION"):
        return True
    linux = sys.platform.startswith("linux")
    return linux and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def open_when_ready(url: str, health: str) -> None:
    # wait until the api is up
    for _ in range(60):
        try:
            urllib.request.urlopen(health, timeout=1)
        except urllib.error.HTTPError:
            pass  # it answers, it just wants the token
        except Exception:
            time.sleep(0.5)
            continue
        webbrowser.open(url)
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="hot-reloading frontend")
    parser.add_argument("--no-open", action="store_true", help="don't open a browser")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the interface")
    parser.add_argument("--host", default=HOST, help="address to listen on")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if args.dev and not loopback(args.host):
        # the dev interface can't pass a token, so others on the network would get in
        raise SystemExit("--dev only works on this computer. Leave out --host, or leave out --dev.")

    sys.path.insert(0, str(SERVER))
    migrate()

    from forge.config import get_settings

    settings = get_settings()
    print(f"data lives in {settings.data_dir}")

    remote = headless()
    if args.dev:
        build_needed = not (WEB / "node_modules").is_dir()
        if build_needed:
            run([npm(), "install"], cwd=WEB)
        vite = subprocess.Popen([npm(), "run", "dev"], cwd=WEB)
        url = "http://localhost:5173"
    else:
        build_interface(force=args.rebuild)
        vite = None
        # through the ssh tunnel the server is on the laptop's own localhost
        name = "localhost" if remote or loopback(args.host) else args.host
        if name in ("0.0.0.0", "::"):
            name = socket.gethostname()
        url = f"http://{name}:{args.port}"
        if remote or not loopback(args.host):
            # only the person with this link gets in
            token = secrets.token_urlsafe(24)
            os.environ["FORGE_TOKEN"] = token
            url = f"{url}/?token={token}"

    port = 5173 if args.dev else args.port
    if remote:
        where = f"{getpass.getuser()}@{socket.gethostname()}"
        print("\nThis computer has no screen. On your laptop, run:")
        print(f"  ssh -L {port}:localhost:{port} {where}")
        print("then open this link there:")
    elif not args.no_open:
        health = f"http://localhost:{port}/api/health"
        threading.Thread(target=open_when_ready, args=(url, health), daemon=True).start()

    print(f"\nForge is running at {url}   (ctrl-c to stop)\n")
    try:
        os.chdir(SERVER)
        import uvicorn

        uvicorn.run("forge.main:app", host=args.host, port=args.port, log_level="warning")
    finally:
        if vite is not None:
            vite.terminate()


if __name__ == "__main__":
    main()
