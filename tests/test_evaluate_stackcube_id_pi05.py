import ast
from pathlib import Path


def test_stackcube_evaluator_exposes_controlled_ood_split():
    path = Path(__file__).resolve().parents[1] / "tools" / "evaluate_stackcube_id_pi05.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    text = ast.unparse(tree)
    assert '"--split"' in text
    assert '"ood"' in text
    assert "STACK_CUBE_OOD_ENV_ID" in text
