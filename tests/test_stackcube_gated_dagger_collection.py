import importlib.util
from pathlib import Path
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


def test_only_complete_ten_step_expert_suffix_is_admitted():
    suffix = MODULE.ExpertSuffix(start=50, action_count=29)
    assert suffix.trainable_chunks == 2
    assert MODULE.selected_suffix_steps(suffix, remaining_chunks=1) == 10
    assert MODULE.selected_suffix_steps(suffix, remaining_chunks=5) == 20
    assert MODULE.selected_suffix_steps(MODULE.ExpertSuffix(None, 0), 2) == 0
