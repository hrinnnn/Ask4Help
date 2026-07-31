from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path


PATH = Path(__file__).parents[1] / "tools" / "libero_plus_failure" / "download_official_assets.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_official_assets", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_historic_official_prefix_is_stripped_during_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "assets.zip"
    prefix = "inspire/hdd/LIBERO-plus-0/assets/"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(prefix + "scenes/table.xml", "scene")
        bundle.writestr(prefix + "README", "asset")
    target = tmp_path / "target" / "assets"
    with zipfile.ZipFile(archive) as bundle:
        actual = MODULE.extract_assets(bundle, target)
    assert actual == prefix
    assert (target / "scenes" / "table.xml").read_text(encoding="utf-8") == "scene"
    assert not (tmp_path / "target" / "inspire").exists()


def test_assets_prefix_rejects_multiple_assets_directories() -> None:
    try:
        MODULE.asset_member_prefix(["a/assets/x", "b/assets/y"])
    except ValueError as error:
        assert "uniquely" in str(error)
    else:
        raise AssertionError("ambiguous archive should fail")
