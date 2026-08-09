from __future__ import annotations

from tools.summarize_xvla_airplane_failure_detection import policy_success


def test_policy_success_supports_airplane_and_stackcube_outcomes() -> None:
    assert policy_success({"ever_grasped": True, "strict_success": False})
    assert not policy_success({"ever_grasped": False, "strict_success": True})
    assert policy_success({"success": True})
    assert not policy_success({"success": False})
