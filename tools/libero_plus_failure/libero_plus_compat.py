"""Narrow compatibility guards for the selected official LIBERO-Plus axes."""

from __future__ import annotations

import os
import sys
import types


WAND_DISABLE_ENV = "LIBERO_PLUS_DISABLE_UNUSED_WAND"


class _UnsupportedMotionBlur:
    argtypes = None

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "official LIBERO-Plus motion blur was requested while " + WAND_DISABLE_ENV + "=1; "
            "this benchmark runner only permits Camera Viewpoints, Robot Initial States, and Objects Layout"
        )


class _UnsupportedWandImage:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "official LIBERO-Plus motion blur was requested while " + WAND_DISABLE_ENV + "=1; "
            "the selected benchmark categories must not use Wand"
        )


def install_unused_wand_stub() -> bool:
    """Install a fail-closed Wand stub only for selected non-image axes.

    Official LIBERO-Plus imports Wand at module import time although Wand is
    only used by its motion-blur image-corruption implementation.  On a
    non-root client host ImageMagick may be unavailable or ABI-incompatible.
    The three frozen leaderboard categories never call that implementation.
    This explicit opt-in stub preserves their semantics and fails loudly if a
    future configuration unexpectedly invokes motion blur.
    """

    if os.environ.get(WAND_DISABLE_ENV) != "1":
        return False
    if "wand" in sys.modules:
        raise RuntimeError(WAND_DISABLE_ENV + " must be set before importing official LIBERO-Plus")
    package = types.ModuleType("wand")
    api = types.ModuleType("wand.api")
    image = types.ModuleType("wand.image")
    api.library = types.SimpleNamespace(MagickMotionBlurImage=_UnsupportedMotionBlur())
    image.Image = _UnsupportedWandImage
    package.api = api
    package.image = image
    sys.modules.update({"wand": package, "wand.api": api, "wand.image": image})
    return True
