from __future__ import annotations

import ast
import json
from pathlib import Path


def _code_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def test_kaggle_notebook_code_cells_are_plain_valid_python():
    path = Path("notebooks/SAM_WM_KAGGLE.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells

    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        assert not source.lstrip().startswith(("!", "%")), f"IPython-only cell {index}"
        ast.parse(source, filename=f"{path}::cell-{index}")


def test_kaggle_notebook_is_single_samwm_pipeline_and_freezes_before_heldout():
    path = Path("notebooks/SAM_WM_KAGGLE.ipynb")
    source = _code_source(path)

    train_suite = source.index('"research.py"')
    preselect = source.index('"promote.py", "preselect"')
    freeze = source.index("FREEZE_MANIFEST.json")
    final_test = source.index('"--open-heldout"')
    finalize = source.index('"promote.py", "finalize"')

    assert train_suite < preselect < freeze < final_test < finalize
    assert "SEEDS = (17, 29, 42, 73, 101)" in source
    assert "SAM_WM_PRE_FREEZE_V2" in source
    assert "GITHUB_TOKEN" in source
    assert "artifacts/research/seed_" in source
    assert "novisad" in source
    assert "fairurbtemp" in source
    assert "no_mental_map" not in source
    assert "unconstrained_exchange" not in source
    assert "no_sigreg" not in source
    assert "temperature_only" not in source
    assert "linear_trend" not in source
    assert "daily_persistence" not in source
    assert "SAM_WM_V41_KAGGLE_INPUT.zip" not in source
