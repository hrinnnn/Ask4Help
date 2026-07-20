from __future__ import annotations

from pathlib import Path


def test_member_sft_uses_an_absolute_rlinf_training_entrypoint():
    script = (Path(__file__).parents[1] / "scripts" / "online_awbc_plug" / "run_member_sft.sh").read_text(
        encoding="utf-8"
    )
    assert '"${PYTHON}" "${RLINF_ROOT}/examples/sft/train_vla_sft.py"' in script
    assert '"${PYTHON}" examples/sft/train_vla_sft.py' not in script


def test_member_sft_exports_an_explicit_single_gpu_rlinf_placement():
    script = (Path(__file__).parents[1] / "scripts" / "online_awbc_plug" / "run_member_sft.sh").read_text(
        encoding="utf-8"
    )
    assert 'export ASK4HELP_RLINF_PLACEMENT="${GPU_ID}-${GPU_ID}"' in script


def test_member_sft_can_use_an_external_two_gpu_ray_head_without_stopping_it():
    script = (Path(__file__).parents[1] / "scripts" / "online_awbc_plug" / "run_member_sft.sh").read_text(
        encoding="utf-8"
    )
    assert 'EXTERNAL_RAY=${EXTERNAL_RAY:-0}' in script
    assert 'if [ "${EXTERNAL_RAY}" = "1" ]; then' in script
    assert 'unset CUDA_VISIBLE_DEVICES' in script


def test_rollout_scripts_require_and_forward_the_pi05_base_directory():
    root = Path(__file__).parents[1] / "scripts" / "online_awbc_plug"
    for name in ("evaluate_checkpoint.sh", "calibrate_fixed_threshold.sh", "collect_ood_round.sh"):
        script = (root / name).read_text(encoding="utf-8")
        assert 'PI05_BASE=${PI05_BASE:?Set PI05_BASE}' in script
        assert '--pi05-base "${PI05_BASE}"' in script


def test_id_checkpoint_grid_keeps_each_policy_evaluation_on_one_gpu():
    script = (
        Path(__file__).parents[1] / "scripts" / "online_awbc_plug" / "evaluate_id_checkpoint_grid.sh"
    ).read_text(encoding="utf-8")
    assert 'POLICY_MEMBER=${POLICY_MEMBER:?Set POLICY_MEMBER to 0 or 1}' in script
    assert 'CUDA_VISIBLE_DEVICES="${GPU_ID}"' in script
    assert 'MEMBER_0="${FIRST_ROOT}/global_step_${step}"' in script
