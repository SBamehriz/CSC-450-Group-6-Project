import builtins
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from forge.gpu import free_gpu_memory


def test_cleanup_when_torch_is_not_installed(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("PyTorch not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Cleanup should exit safely if PyTorch is missing.
    free_gpu_memory()


def test_cleanup_when_cuda_is_unavailable(monkeypatch):
    empty_cache = MagicMock()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            empty_cache=empty_cache,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    free_gpu_memory()

    empty_cache.assert_not_called()


def test_cleanup_when_cuda_is_available(monkeypatch):
    empty_cache = MagicMock()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=empty_cache,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    free_gpu_memory()

    empty_cache.assert_called_once_with()
