from __future__ import annotations

import ast
import json
from pathlib import Path


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


def test_kaggle_notebook_has_research_suite_and_freeze_before_heldout():
    path = Path("notebooks/SAM_WM_KAGGLE.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    research_stage = source.index('"research.py"')
    freeze = source.index("FREEZE_MANIFEST.json")
    final_test = source.index('"--open-heldout"')
    assert research_stage < freeze < final_test
    assert "SEEDS = (17, 29, 42, 73, 101)" in source
    assert "PRE_FREEZE_MANIFEST.json" in source
    assert "GITHUB_TOKEN" in source
    assert "SAM_WM_V41_KAGGLE_INPUT.zip" not in source
