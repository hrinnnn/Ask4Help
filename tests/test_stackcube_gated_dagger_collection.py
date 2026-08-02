import importlib.util
import hashlib
from pathlib import Path
import subprocess
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "collect_stackcube_gated_dagger.py"
SPEC = importlib.util.spec_from_file_location("stackcube_gated_dagger_collection", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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
    assert MODULE.should_latch_expert(
        "diffdagger", action_step=10, score=1.0, threshold=1.0
    )
    assert not MODULE.should_latch_expert("failure_recovery", action_step=100)


def test_only_complete_ten_step_expert_suffix_is_admitted():
    suffix = MODULE.ExpertSuffix(start=50, action_count=29)
    assert suffix.trainable_chunks == 2
    assert suffix.valid_10_step_anchors == 20
    assert MODULE.selected_suffix_steps(suffix, remaining_chunks=1) == 10
    assert MODULE.selected_suffix_steps(suffix, remaining_chunks=5) == 20
    assert MODULE.selected_suffix_steps(MODULE.ExpertSuffix(None, 0), 2) == 0


def test_fixed_rollout_collection_keeps_all_complete_expert_suffix_chunks():
    suffix = MODULE.ExpertSuffix(start=50, action_count=29)
    assert MODULE.admitted_suffix_steps(
        suffix, remaining_chunks=1, fixed_episode_collection=True
    ) == 29
    assert MODULE.admitted_suffix_steps(
        suffix, remaining_chunks=1, fixed_episode_collection=False
    ) == 10
    assert MODULE.admitted_suffix_steps(
        suffix, remaining_chunks=1, fixed_episode_collection=True,
        preserve_terminal_suffix=True,
    ) == 29


def test_failure_recovery_oracle_only_latches_after_a_recorded_predicate():
    state = MODULE.FailureRecoveryState(initial_height=0.02, best_goal_distance=0.1)
    assert state.stagnant_chunks == 0
    short = MODULE.ExpertSuffix(start=50, action_count=9)
    assert short.valid_10_step_anchors == 0
    assert MODULE.admitted_suffix_steps(
        short, remaining_chunks=1, fixed_episode_collection=True
    ) == 0


def test_bridge_knn_uses_persisted_k10_detector_name():
    detector = object()
    resolved = MODULE._resolve_detector(
        "bridge_knn",
        {"vlm_bridge_final_mean__knn_k10": detector},
        {"detectors": {"vlm_bridge_final_mean__knn_k10": {"threshold": 3.5}}},
    )
    assert resolved == ("vlm_bridge_final_mean__knn_k10", detector, 3.5)


def test_policy_provenance_requires_and_hashes_the_exact_checkpoint(tmp_path):
    checkpoint = tmp_path / "global_step_7000"
    weights = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"immutable policy")
    pi05_base = tmp_path / "pi05_base"
    pi05_base.mkdir()
    norm_stats = tmp_path / "norm_stats.json"
    norm_stats.write_text("{}", encoding="utf-8")

    provenance = MODULE.policy_provenance(
        method="failure_recovery",
        checkpoint=checkpoint,
        pi05_base=pi05_base,
        norm_stats=norm_stats,
    )

    assert provenance["policy_checkpoint_name"] == "global_step_7000"
    assert provenance["policy_full_weights_sha256"] == hashlib.sha256(b"immutable policy").hexdigest()
    assert provenance["norm_stats_sha256"] == hashlib.sha256(b"{}").hexdigest()


def test_group_bc_launcher_has_valid_shell_syntax():
    launcher = MODULE_PATH.parents[1] / "scripts" / "stackcube_gated_dagger" / "run_group_bc.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
