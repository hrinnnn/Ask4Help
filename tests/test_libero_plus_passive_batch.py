from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PATH = Path(__file__).parents[1] / "tools" / "libero_plus_failure" / "run_passive_batch.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_passive_batch", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_calibration_schedule_is_balanced_and_reproducible() -> None:
    rows = MODULE.calibration_schedule(task_count=3, max_attempts=8, seed_base=10)
    assert [row["task_index"] for row in rows] == [0, 1, 2, 0, 1, 2, 0, 1]
    assert [row["seed"] for row in rows] == list(range(10, 18))
    assert MODULE.episode_dir(Path("out"), rows[0]).name == "attempt_00000_task_00_seed_000010"


def test_manifest_schedule_preserves_every_official_configuration_and_pairing() -> None:
    manifest = {"rows": [
        {"plus_task_id": 11, "plus_task_index": 10, "clean_task_index": 1, "category": "Camera Viewpoints"},
        {"plus_task_id": 12, "plus_task_index": 11, "clean_task_index": 2, "category": "Objects Layout"},
    ]}
    plus = MODULE.manifest_schedule(manifest, task_field="plus_task_index")
    clean = MODULE.manifest_schedule(manifest, task_field="clean_task_index")
    assert [row["task_index"] for row in plus] == [10, 11]
    assert [row["task_index"] for row in clean] == [1, 2]
    assert [row["seed"] for row in plus] == [200011, 200012]
    assert MODULE.episode_dir(Path("out"), plus[0]).name == "config_000011"


def test_completed_rollout_requires_all_immutable_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "episode"
    target.mkdir()
    (target / "episode.json").write_text(json.dumps({"success": True, "timeline": [{}]}), encoding="utf-8")
    assert not MODULE.completed_rollout(target)
    (target / "features.npz").write_bytes(b"features")
    (target / "rollout.mp4").write_bytes(b"video")
    assert MODULE.completed_rollout(target)


def test_task_balanced_calibration_can_count_successes_independently(tmp_path: Path) -> None:
    rows = MODULE.calibration_schedule(task_count=2, max_attempts=4, seed_base=1)
    for row in rows[:2]:
        path = MODULE.episode_dir(tmp_path, row)
        path.mkdir(parents=True)
        (path / "episode.json").write_text(json.dumps({"success": True, "timeline": [{}]}), encoding="utf-8")
        (path / "features.npz").write_bytes(b"x")
        (path / "rollout.mp4").write_bytes(b"x")
    assert MODULE.completed_successes_by_task(rows, tmp_path) == {0: 1, 1: 1}
