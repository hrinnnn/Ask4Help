import json

from tools.build_stackcube_xvla_timing_intersection import common_seeds


def test_intersection_preserves_immediate_seed_order(tmp_path) -> None:
    methods = ("immediate", "post_grasp", "post_lift", "failure_recovery")
    rows = {
        "immediate": [3, 1, 2], "post_grasp": [1, 2, 3],
        "post_lift": [3, 1], "failure_recovery": [1, 3],
    }
    for method in methods:
        path = tmp_path / method
        path.mkdir()
        (path / "training_episodes.jsonl").write_text(
            "".join(json.dumps({"seed": seed}) + "\n" for seed in rows[method])
        )
    assert common_seeds(tmp_path) == [3, 1]
