import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "start.py"
# what uvicorn was asked to serve
SERVED: dict = {}


@pytest.fixture()
def start(monkeypatch):
    spec = importlib.util.spec_from_file_location("forge_start_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import uvicorn

    served = SERVED
    served.clear()
    monkeypatch.setattr(module, "migrate", lambda: None)
    monkeypatch.setattr(module, "build_interface", lambda force=False: None)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: served.update(kwargs))
    monkeypatch.chdir(os.getcwd())
    # main sets it; this puts back what was there after the test
    monkeypatch.setenv("FORGE_TOKEN", "")
    monkeypatch.delenv("FORGE_TOKEN")
    for name in ("SSH_CONNECTION", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    return module


def launch(start, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["start.py", *argv])
    start.main()
    return SERVED


def test_loopback_and_screens_are_told_apart(start, monkeypatch):
    assert start.loopback("127.0.0.1") and start.loopback("localhost") and start.loopback("::1")
    assert not start.loopback("0.0.0.0") and not start.loopback("lab-server")
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.2 5000 10.0.0.1 22")
    assert start.headless()
    monkeypatch.delenv("SSH_CONNECTION")
    monkeypatch.setattr(sys, "platform", "linux")
    assert start.headless()
    monkeypatch.setenv("DISPLAY", ":0")
    assert not start.headless()


def test_over_ssh_it_prints_the_tunnel_and_a_link_with_a_token(start, monkeypatch, capsys):
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.2 5000 10.0.0.1 22")
    opened = []
    monkeypatch.setattr(start.webbrowser, "open", opened.append)
    served = launch(start, monkeypatch, "--port", "8123")
    said = capsys.readouterr().out
    assert "ssh -L 8123:localhost:8123 " in said
    token = os.environ["FORGE_TOKEN"]
    assert len(token) > 20 and f"http://localhost:8123/?token={token}" in said
    assert served == {"host": "127.0.0.1", "port": 8123, "log_level": "warning"}
    assert opened == []


def test_another_host_gets_a_token_and_dev_stays_on_this_computer(start, monkeypatch, capsys):
    served = launch(start, monkeypatch, "--host", "0.0.0.0", "--no-open")
    assert served["host"] == "0.0.0.0"
    assert f"?token={os.environ['FORGE_TOKEN']}" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--dev only works on this computer"):
        launch(start, monkeypatch, "--dev", "--host", "0.0.0.0")


def test_on_this_computer_there_is_no_token(start, monkeypatch, capsys):
    # a screen, or linux with none (like the test runner) counts as a server
    monkeypatch.setenv("DISPLAY", ":0")
    launch(start, monkeypatch, "--no-open")
    assert "FORGE_TOKEN" not in os.environ
    assert "http://localhost:8000 " in capsys.readouterr().out


def test_without_npm_it_says_how_to_get_the_interface(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("forge_start_build", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WEB", tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="copy the web/dist folder"):
        module.build_interface()
