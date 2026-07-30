import ast
from pathlib import Path


def test_stackcube_evaluator_exposes_controlled_ood_split():
    path = Path(__file__).resolve().parents[1] / "tools" / "evaluate_stackcube_id_pi05.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    text = ast.unparse(tree)
    string_literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert {"--split", "ood"} <= string_literals
    assert "STACK_CUBE_OOD_ENV_ID" in text
