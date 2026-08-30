"""Optional per-process Ray object-store cap for the adaptive timing jobs.

The upstream Ray default is about 200 GiB on this host.  Four independent
single-GPU jobs would therefore overrun the 504 GiB shared-memory mount.  The
shim is enabled only when ``ASK4HELP_RAY_OBJECT_STORE_MEMORY`` is set and
leaves connections to an existing Ray cluster untouched.
"""

from __future__ import annotations

import os


def _install() -> None:
    raw = os.environ.get("ASK4HELP_RAY_OBJECT_STORE_MEMORY")
    if not raw:
        return
    try:
        cap = int(raw)
    except ValueError:
        return
    try:
        import ray
    except Exception:
        return
    if getattr(ray.init, "_ask4help_object_store_cap", False):
        return
    original = ray.init

    def init(*args, **kwargs):
        address = kwargs.get("address")
        if address is None and args:
            address = args[0]
        connects_existing = isinstance(address, str) and (
            address == "auto" or address.startswith("ray://")
        )
        if not connects_existing and kwargs.get("object_store_memory") is None:
            kwargs["object_store_memory"] = cap
        return original(*args, **kwargs)

    init._ask4help_object_store_cap = True
    ray.init = init


_install()
