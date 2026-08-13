from pathlib import Path


def test_uncover_sphere_place_is_registered_as_an_auditable_candidate() -> None:
    module = Path("RLinf/rlinf/envs/maniskill/uncover_sphere_place.py")
    spec = Path("docs/experiment_management/plans/UncoverSpherePlace_ID_and_stage_OOD.md")
    assert module.exists()
    assert spec.exists()
    text = module.read_text()
    assert 'UNCOVER_ENV_IDS' in text
    assert 'handle_ood' in text
    assert 'goal_ood' in text
    assert 'success' in text


def test_uncover_sphere_place_changes_one_stage_factor_at_a_time() -> None:
    text = Path("docs/experiment_management/plans/UncoverSpherePlace_ID_and_stage_OOD.md").read_text()
    assert "只改变遮挡物 yaw" in text
    assert "只镜像目标碗位置" in text
    assert "paired reset" in text
