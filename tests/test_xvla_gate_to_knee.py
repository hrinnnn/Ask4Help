from tools.summarize_xvla_gate_to_knee import summarize_method


def test_gate_to_knee_reports_miss_and_hit_rates():
    rows = [
        {
            "episode_index": 0,
            "seed": 10,
            "timeline": [{"env_step": 5, "scores": {"diffdagger": 0.2}}],
        },
        {
            "episode_index": 1,
            "seed": 11,
            "timeline": [{"env_step": 20, "scores": {"diffdagger": 0.2}}],
        },
        {
            "episode_index": 2,
            "seed": 12,
            "timeline": [{"env_step": 5, "scores": {"diffdagger": 0.1}}],
        },
    ]
    result = summarize_method(
        rows,
        method="diffdagger",
        knee_set=[20],
        threshold=0.15,
        knee_tolerance=5,
    )
    assert result["alarms_observed"] == 2
    assert result["alarm_miss_rate"] == 1 / 3
    assert result["knee_hit_rate_conditional"] == 0.5
    assert result["knee_distance_mean"] == 7.5


def test_fixed_failure_recovery_step_is_a_gate_time():
    result = summarize_method(
        [{"episode_index": 0, "seed": 1, "timeline": []}],
        method="failure_recovery",
        knee_set=[45],
        threshold=None,
        knee_tolerance=5,
        fixed_step=50,
    )
    assert result["alarms_observed"] == 1
    assert result["knee_distance_mean"] == 5
    assert result["knee_hit_rate_all_episodes"] == 1.0
