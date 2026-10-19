import importlib.util
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TOOL: tuple[str, ...] = ("nvidia-smi",)
FORMAT = "--format=csv,noheader,nounits"
CARD_QUERY = "--query-gpu=index,uuid,name,memory.total,driver_version,compute_cap"
ASK_FOR_S = 5.0
SMALLEST_MIB, BIGGEST_MIB = 512, 1 << 20

TorchBuild = Literal["cuda", "rocm", "cpu", "missing", "unknown"]


@dataclass(frozen=True)
class Card:
    index: int
    name: str
    memory_bytes: int
    driver: str
    compute_capability: str


@dataclass(frozen=True)
class Reading:
    cards: list[Card]
    said: str | None
    torch: TorchBuild
    torch_version: str | None
    torch_cuda: str | None


def _ask(*query: str) -> tuple[list[str] | None, str | None]:
    try:
        asked = subprocess.Popen(
            [*TOOL, *query, FORMAT], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except OSError:
        return None, None
    try:
        out, err = asked.communicate(timeout=ASK_FOR_S)
    except subprocess.TimeoutExpired:
        asked.kill()
        return None, None
    said = out.decode("utf-8", "replace").strip()
    if asked.returncode != 0:
        words = [part for part in (said, err.decode("utf-8", "replace").strip()) if part]
        return None, "\n".join(words) or None
    return [line.strip() for line in said.splitlines() if line.strip()], None


def _whole(text: str, most: int) -> int | None:
    return int(text) if text.isascii() and text.isdecimal() and int(text) <= most else None


def _a_card(line: str) -> Card | None:
    front = [part.strip() for part in line.split(",", 2)]
    if len(front) != 3:
        return None
    index, uuid, rest = front
    parts = [part.strip() for part in rest.rsplit(",", 3)]
    if _whole(index, 1 << 16) is None or not uuid.startswith("GPU-") or len(parts) != 4:
        return None
    name, mib, driver, capability = parts
    total = _whole(mib, BIGGEST_MIB)
    if (
        not name
        or total is None
        or total < SMALLEST_MIB
        or not re.fullmatch(r"[0-9]+(\.[0-9]+)*", driver)
        or not re.fullmatch(r"[0-9]+\.[0-9]+", capability)
    ):
        return None
    return Card(int(index), name, total * 2**20, driver, capability)


def _cards() -> tuple[list[Card], str | None]:
    lines, said = _ask(CARD_QUERY)
    if lines is None:
        return [], said
    cards = [card for card in map(_a_card, lines) if card is not None]
    if not cards:
        return [], "\n".join(lines) or None
    return cards, None


def _torch_said() -> str | None:
    try:
        found = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if found is None or not found.submodule_search_locations:
        return None
    try:
        folder = Path(next(iter(found.submodule_search_locations)))
        return (folder / "version.py").read_text(encoding="utf-8")
    except (OSError, StopIteration, UnicodeDecodeError):
        return ""


def _value(said: str, name: str) -> str | None:
    found = re.search(rf"^{name}\s*(?::[^=\n]*)?=\s*(['\"])([^'\"\n]*)\1", said, re.M)
    return found.group(2) if found else None


def _torch(said: str | None) -> tuple[TorchBuild, str | None, str | None]:
    if said is None:
        return "missing", None, None
    version = _value(said, "__version__")
    cuda = _value(said, "cuda")
    if cuda:
        return "cuda", version, cuda
    if _value(said, "hip"):
        return "rocm", version, None
    return ("cpu", version, None) if version else ("unknown", None, None)


def read() -> Reading:
    cards, said = _cards()
    return Reading(cards, said, *_torch(_torch_said()))
