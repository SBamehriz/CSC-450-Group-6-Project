import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "split_text.py"


def load():
    spec = importlib.util.spec_from_file_location("split_text", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pieces_hold_every_line_once_and_end_at_line_ends(tmp_path):
    source = tmp_path / "big.txt"
    lines = [f"Once upon a time there was story number {index}.\n" for index in range(2000)]
    source.write_text("".join(lines), encoding="utf-8")
    count = load().split(source, tmp_path / "pieces", size_bytes=10_000)
    files = sorted((tmp_path / "pieces").glob("big-*.txt"))
    assert count == len(files) > 5
    joined = "".join(path.read_text(encoding="utf-8") for path in files)
    assert joined == source.read_text(encoding="utf-8")
    assert all(path.read_text(encoding="utf-8").endswith("\n") for path in files)
