import importlib.util
from pathlib import Path
import subprocess
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "collect_stackcube_gated_dagger.py"
SPEC = importlib.util.spec_from_file_location("stackcube_gated_dagger_collection", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

REBUILD_PATH = Path(__file__).resolve().parents[1] / "tools" / "rebuild_stackcube_gated_dagger_dataset.py"
REBUILD_SPEC = importlib.util.spec_from_file_location("stackcube_gated_dagger_rebuild", REBUILD_PATH)
REBUILD = importlib.util.module_from_spec(REBUILD_SPEC)
assert REBUILD_SPEC.loader is not None
sys.modules[REBUILD_SPEC.name] = REBUILD
REBUILD_SPEC.loader.exec_module(REBUILD)

NORM_PATH = Path(__file__).resolve().parents[1] / "tools" / "calculate_source_balanced_openpi_norm_stats.py"
NORM_SPEC = importlib.util.spec_from_file_location("stackcube_gated_dagger_norm", NORM_PATH)
NORM = importlib.util.module_from_spec(NORM_SPEC)
assert NORM_SPEC.loader is not None
sys.modules[NORM_SPEC.name] = NORM
NORM_SPEC.loader.exec_module(NORM)


def test_quota_scheduler_prefers_remaining_label_budget_and_alternates_ties():
    assert MODULE.choose_split({"id": 0, "ood": 0}, {"id": 3, "ood": 3}, prefer_id=True) == "id"
    assert MODULE.choose_split({"id": 3, "ood": 1}, {"id": 3, "ood": 3}, prefer_id=True) == "ood"
    assert MODULE.choose_split({"id": 3, "ood": 3}, {"id": 3, "ood": 3}, prefer_id=True) is None


def test_successful_expert_collection_alternates_raw_splits_and_filters_labels():
    assert [MODULE.alternating_split(index) for index in range(6)] == [
        "id", "ood", "id", "ood", "id", "ood"
    ]
    assert MODULE.is_successful_expert_trajectory(
        MODULE.ExpertSuffix(start=50, action_count=10), success=True
    )
    assert not MODULE.is_successful_expert_trajectory(
        MODULE.ExpertSuffix(start=50, action_count=9), success=True
    )
    assert not MODULE.is_successful_expert_trajectory(
        MODULE.ExpertSuffix(start=50, action_count=10), success=False
    )


def test_collection_gate_semantics_are_latched_at_the_specified_boundary():
    assert MODULE.should_latch_expert("offline_oracle", action_step=0)
    assert not MODULE.should_latch_expert("late_success", action_step=45)
    assert MODULE.should_latch_expert("late_success", action_step=50)
    assert not MODULE.should_latch_expert(
        "bridge_knn", action_step=10, score=0.9, threshold=1.0
    )
    assert MODULE.should_latch_expert(
        "bridge_knn", action_step=10, score=1.0, threshold=1.0
    )


def test_successful_suffix_retains_terminal_actions_and_counts_valid_anchors():
    suffix = MODULE.ExpertSuffix(start=50, action_count=29)
    assert suffix.trainable_chunks == 2
    assert suffix.valid_10_step_anchors == 20
    assert suffix.has_full_horizon
    assert MODULE.selected_suffix_steps(suffix, remaining_chunks=1) == 10
    assert MODULE.selected_suffix_steps(suffix, remaining_chunks=5) == 20
    assert MODULE.selected_suffix_steps(MODULE.ExpertSuffix(None, 0), 2) == 0


def test_fixed_rollout_collection_keeps_every_terminal_expert_action():
    suffix = MODULE.ExpertSuffix(start=50, action_count=29)
    assert MODULE.admitted_suffix_steps(
        suffix, remaining_chunks=1, fixed_episode_collection=True
    ) == 29
    assert MODULE.admitted_suffix_steps(
        suffix, remaining_chunks=1, fixed_episode_collection=False
    ) == 10


def test_suffix_shorter_than_horizon_is_not_admitted():
    suffix = MODULE.ExpertSuffix(start=50, action_count=9)
    assert suffix.valid_10_step_anchors == 0
    assert not suffix.has_full_horizon
    assert MODULE.admitted_suffix_steps(
        suffix, remaining_chunks=10, fixed_episode_collection=True
    ) == 0


def test_bridge_knn_uses_persisted_k10_detector_name():
    detector = object()
    resolved = MODULE._resolve_detector(
        "bridge_knn",
        {"vlm_bridge_final_mean__knn_k10": detector},
        {"detectors": {"vlm_bridge_final_mean__knn_k10": {"threshold": 3.5}}},
    )
    assert resolved == ("vlm_bridge_final_mean__knn_k10", detector, 3.5)


def test_rebuild_resolves_legacy_row_to_full_terminal_expert_suffix():
    episodes = [
        {
            "episode_index": 7,
            "success": True,
            "expert_start_step": 50,
            "expert_action_steps": 26,
        }
    ]
    selected = REBUILD.select_complete_successful_suffixes(
        episodes, [{"raw_episode_index": 7, "action_steps": 20}]
    )
    assert selected[0]["expert_action_steps"] == 26


def test_rebuild_rejects_failed_or_incomplete_legacy_suffixes():
    for row in (
        {"episode_index": 1, "success": False, "expert_start_step": 0, "expert_action_steps": 20},
        {"episode_index": 2, "success": True, "expert_start_step": 0, "expert_action_steps": 9},
    ):
        try:
            REBUILD.select_complete_successful_suffixes([row], [{"raw_episode_index": row["episode_index"]}])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid suffix was accepted")


def test_rebuild_retries_contact_jitter_but_never_accepts_a_failed_replay(monkeypatch):
    outcomes = iter(
        [(["failed-records"], False), (["success-records"], True)]
    )
    monkeypatch.setattr(REBUILD, "_replay_full_episode", lambda *_args: next(outcomes))
    records, attempts = REBUILD.replay_until_success(
        object(), {"episode_index": 3}, None, max_attempts=3
    )
    assert records == ["success-records"]
    assert attempts == 2


def test_valid_action_anchor_indices_exclude_exactly_the_episode_tail():
    indices = NORM.valid_anchor_indices([(0, 20), (20, 50)], horizon=10)
    assert indices.tolist() == list(range(0, 11)) + list(range(20, 41))


def test_group_bc_launcher_has_valid_shell_syntax():
    launcher = MODULE_PATH.parents[1] / "scripts" / "stackcube_gated_dagger" / "run_group_bc.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
