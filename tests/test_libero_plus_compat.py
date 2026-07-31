from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "tools" / "libero_plus_failure" / "libero_plus_compat.py"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, PATH)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_wand_stub_is_explicit_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIBERO_PLUS_DISABLE_UNUSED_WAND", "1")
    for name in ("wand", "wand.api", "wand.image"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = _module("libero_plus_compat_test")
    assert module.install_unused_wand_stub()
    from wand.api import library
    from wand.image import Image
    with pytest.raises(RuntimeError, match="motion blur"):
        library.MagickMotionBlurImage(None)
    with pytest.raises(RuntimeError, match="motion blur"):
        Image()


def test_wand_stub_is_off_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIBERO_PLUS_DISABLE_UNUSED_WAND", raising=False)
    module = _module("libero_plus_compat_off_test")
    assert not module.install_unused_wand_stub()
