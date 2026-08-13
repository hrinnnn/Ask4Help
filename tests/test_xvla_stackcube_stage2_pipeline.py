from tools.run_xvla_stackcube_stage2_pipeline import METHODS


def test_stage2_pipeline_compares_the_four_formal_groups() -> None:
    assert METHODS == (
        "internal_pca",
        "offline_oracle",
        "failure_recovery",
        "diffdagger",
    )
