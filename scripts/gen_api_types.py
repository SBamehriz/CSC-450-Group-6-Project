#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
WEB = ROOT / "web"
OUT = WEB / "src" / "api" / "types.ts"

DUMP_SPEC = """
import json, sys
from forge.main import create_app
json.dump(create_app().openapi(), sys.stdout, indent=2)
"""


def tool(name: str) -> str:
    # windows sometimes uses npx.cmd
    found = shutil.which(name) or shutil.which(f"{name}.cmd")
    if not found:
        raise SystemExit(f"{name} was not found on your PATH.")
    return found


def run(command: list[str], cwd: Path, capture: bool = False) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=capture)
    if result.returncode != 0:
        if capture and result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(command)}")
    return result.stdout if capture else ""


def main() -> None:
    spec = run([sys.executable, "-c", DUMP_SPEC], cwd=SERVER, capture=True)
    # fail here, not inside npx
    json.loads(spec)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        handle.write(spec)
        spec_path = handle.name

    try:
        npx = tool("npx")
        run([npx, "--yes", "openapi-typescript", spec_path, "--output", str(OUT)], cwd=WEB)
        run([npx, "--yes", "prettier", "--write", str(OUT)], cwd=WEB)
    finally:
        os.unlink(spec_path)

    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
