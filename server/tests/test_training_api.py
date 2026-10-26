import sys

import pytest

from forge import gpu
from forge.train.presets import PRESETS, training_memory

CARDS = (
    "0, GPU-1111aaaa, NVIDIA GeForce RTX 3090, 24576, 595.79, 8.6\n"
    "1, GPU-2222bbbb, NVIDIA A100-SXM4-80GB, 81920, 580.12, 8.0\n"
)


def stand_in(monkeypatch, tmp_path, printed: str, code: int = 0, err: str = ""):
    script = tmp_path / "nvidia_smi.py"
    script.write_text(
        f"import sys\nsys.stdout.write({printed!r})\nsys.stderr.write({err!r})\nsys.exit({code})\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gpu, "TOOL", (sys.executable, str(script)))


def test_presets_list_every_size_with_its_count_and_memory(client):
    response = client.get("/api/presets")
    assert response.status_code == 200
    sizes = {preset["name"]: preset for preset in response.json()}
    assert list(sizes) == list(PRESETS)
    assert sizes["135m"]["parameters"] == 133_864_704
    assert (sizes["135m"]["n_layer"], sizes["135m"]["d_model"], sizes["135m"]["ctx_len"]) == (
        17,
        768,
        1024,
    )
    for name, preset in sizes.items():
        steps = [row["sequences_per_step"] for row in preset["memory"]]
        assert steps == [1, 2, 4, 8, 16]
        for row in preset["memory"]:
            parts = training_memory(PRESETS[name], micro_batch_seqs=row["sequences_per_step"])
            assert row["bytes"] == sum(parts.values())
        needed = [row["bytes"] for row in preset["memory"]]
        assert needed == sorted(needed)


def test_gpu_check_reports_every_card_the_driver_lists(client, monkeypatch, tmp_path):
    stand_in(monkeypatch, tmp_path, CARDS)
    body = client.get("/api/gpu").json()
    assert body["cards"] == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 3090",
            "memory_bytes": 24576 * 2**20,
            "driver": "595.79",
            "compute_capability": "8.6",
        },
        {
            "index": 1,
            "name": "NVIDIA A100-SXM4-80GB",
            "memory_bytes": 81920 * 2**20,
            "driver": "580.12",
            "compute_capability": "8.0",
        },
    ]
    assert body["tool_said"] is None


def test_gpu_check_with_no_driver_tool_finds_no_card(client, monkeypatch):
    monkeypatch.setattr(gpu, "TOOL", ("forge-no-such-tool",))
    body = client.get("/api/gpu").json()
    assert body["cards"] == []
    assert body["tool_said"] is None


def test_gpu_check_passes_on_what_a_failing_driver_said(client, monkeypatch, tmp_path):
    stand_in(monkeypatch, tmp_path, "", code=9, err="NVIDIA-SMI has failed.")
    body = client.get("/api/gpu").json()
    assert body["cards"] == []
    assert body["tool_said"] == "NVIDIA-SMI has failed."


@pytest.mark.parametrize(
    "line",
    [
        "0, GPU-1111aaaa, Some Card, [N/A], 595.79, 8.6",
        "0, not-a-uuid, Some Card, 8192, 595.79, 8.6",
        "0, GPU-1111aaaa, Some Card, 100, 595.79, 8.6",
        "0, GPU-1111aaaa, , 8192, 595.79, 8.6",
        "No devices were found",
    ],
)
def test_an_answer_that_does_not_add_up_is_not_a_card(line):
    assert gpu._a_card(line) is None


def test_a_card_name_with_a_comma_keeps_its_numbers():
    card = gpu._a_card("2, GPU-3333cccc, Card, Rev 2, 12288, 595.79, 8.9")
    assert card == gpu.Card(2, "Card, Rev 2", 12288 * 2**20, "595.79", "8.9")


@pytest.mark.parametrize(
    "said, expected",
    [
        (None, ("missing", None, None)),
        (
            "__version__ = '2.14.1+cu130'\ncuda: Optional[str] = '13.0'\n",
            ("cuda", "2.14.1+cu130", "13.0"),
        ),
        (
            "__version__ = '2.14.1+rocm6.4'\ncuda = None\nhip = '6.4'\n",
            ("rocm", "2.14.1+rocm6.4", None),
        ),
        ("__version__ = '2.14.1+cpu'\ncuda = None\n", ("cpu", "2.14.1+cpu", None)),
        ("", ("unknown", None, None)),
    ],
)
def test_which_torch_is_installed_is_read_without_importing_it(said, expected):
    assert gpu._torch(said) == expected


def test_gpu_check_says_when_torch_is_missing(client, monkeypatch):
    monkeypatch.setattr(gpu, "TOOL", ("forge-no-such-tool",))
    monkeypatch.setattr(gpu, "_torch_said", lambda: None)
    body = client.get("/api/gpu").json()
    assert (body["torch"], body["torch_version"], body["torch_cuda"]) == ("missing", None, None)
