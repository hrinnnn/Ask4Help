import pytest

from tools import run_xvla_airplane_centered_oracle_pipeline as pipeline


centered_oracle_validation = pipeline.centered_oracle_validation


def _row(accepted: bool, *, centered: bool = True, shift: float = 0.002) -> dict:
    candidate = "neck_center_x_minus_014_y_minus_046_z_plus_010" if centered else "old_offset"
    return {
        "accepted": accepted,
        "oracle": {
            "selected_candidate": candidate,
            "attempts": [
                {
                    "candidate": candidate,
                    "object_xy_shift_before_close": shift * 2,
                    "object_xy_shift_during_close": shift,
                }
            ],
        },
    }


def test_centered_oracle_validation_counts_only_accepted_rows() -> None:
    report = centered_oracle_validation([_row(True), _row(True, shift=0.004), _row(False)])
    assert report == {
        "raw_attempts": 3,
        "accepted": 2,
        "all_centered": True,
        "max_approach_shift_mm": 8.0,
        "max_close_shift_mm": 4.0,
    }


def test_centered_oracle_validation_rejects_offset_candidate() -> None:
    report = centered_oracle_validation([_row(True, centered=False)])
    assert not report["all_centered"]


@pytest.mark.parametrize("returncode,accepted_abort", [(0, False), (-6, True)])
def test_collection_exit_accepts_complete_artifacts(
    monkeypatch: pytest.MonkeyPatch, returncode: int, accepted_abort: bool
) -> None:
    monkeypatch.setattr(pipeline, "validate_collection", lambda method: {"method": method})
    report = pipeline.validate_collection_exit("failure_recovery", returncode)
    assert report["accepted_teardown_abort"] is accepted_abort


def test_collection_exit_rejects_other_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "validate_collection", lambda method: {"method": method})
    with pytest.raises(RuntimeError, match="returncode=1"):
        pipeline.validate_collection_exit("failure_recovery", 1)
