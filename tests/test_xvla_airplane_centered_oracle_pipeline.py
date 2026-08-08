from tools.run_xvla_airplane_centered_oracle_pipeline import centered_oracle_validation


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
