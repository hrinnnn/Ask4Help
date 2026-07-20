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
