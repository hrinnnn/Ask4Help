from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools" / "collect_pick_single_ycb_airplane_gated_dagger.py"
SPEC = importlib.util.spec_from_file_location("airplane_gated", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_raw_attempts_strictly_alternate_id_then_ood():
    assert [MODULE.alternating_split(index) for index in range(6)] == ["id", "ood"] * 3


def test_bridge_pca_uses_strict_threshold_exceedance():
    assert not MODULE.should_query_bridge_pca(0.5, 0.5)
    assert MODULE.should_query_bridge_pca(0.50001, 0.5)


def test_all_real_terminal_expert_actions_are_admitted():
    assert MODULE.admitted_expert_suffix(success=True, expert_start=50, action_count=53) == (50, 53)
    assert MODULE.admitted_expert_suffix(success=True, expert_start=50, action_count=51) == (50, 51)


def test_failed_or_empty_suffix_is_not_admitted():
    assert MODULE.admitted_expert_suffix(success=False, expert_start=50, action_count=60) is None
    assert MODULE.admitted_expert_suffix(success=True, expert_start=None, action_count=60) is None
    assert MODULE.admitted_expert_suffix(success=True, expert_start=60, action_count=60) is None


def test_zero_action_raw_attempt_remains_a_rejected_training_example():
    assert MODULE.admitted_expert_suffix(success=False, expert_start=0, action_count=0) is None


def test_delta_servo_requires_all_arm_joints_within_tolerance():
    target = np.zeros(8, dtype=np.float32)
    current = np.zeros(9, dtype=np.float32)
    assert MODULE.delta_servo_complete(current, target, tolerance=0.012)
    current[3] = 0.011
    assert MODULE.delta_servo_complete(current, target, tolerance=0.012)
    current[3] = 0.013
    assert not MODULE.delta_servo_complete(current, target, tolerance=0.012)


def test_delta_servo_ignores_gripper_target_for_arm_completion():
    target = np.zeros(8, dtype=np.float32)
    target[7] = 1.0
    current = np.zeros(9, dtype=np.float32)
    assert MODULE.delta_servo_complete(current, target, tolerance=0.012)


def test_patience_gate_calibrates_the_same_temporal_event_used_online():
    sequences = [
        [0.1, 0.9, 0.2],  # isolated spike cannot pass patience=2
        [0.1, 0.7, 0.6],  # limiting adjacent score is 0.6
        [0.4, 0.5, 0.1],  # limiting adjacent score is 0.4
    ]
    threshold, episode_scores = MODULE.calibrate_patience_gate_threshold(
        sequences, alpha=0.5, patience=2
    )
    assert episode_scores == [0.2, 0.6, 0.4]
    assert threshold == 0.4


def test_patience_gate_rejects_sequences_shorter_than_its_window():
    assert MODULE.patience_gate_episode_score([0.9], patience=2) == float("-inf")
