from tools.run_xvla_airplane_ood_dagger_pipeline import METHODS


def test_pipeline_has_the_four_locked_groups() -> None:
    assert METHODS == (
        "vlm_pool_pca",
        "offline_oracle",
        "failure_recovery",
        "diffdagger",
    )
