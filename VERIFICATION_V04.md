# SAM-WM v0.4 verification

Executed in the artifact build environment:

- Python `compileall`: **PASS**
- JavaScript syntax (`node --check static/app.js`): **PASS**
- Non-live pytest suite:

```text
...................ss....                                                [100%]
=============================== warnings summary ===============================
tests/test_world_model.py::test_world_model_shapes_and_future_conditioning
  /mnt/data/SAM-WM-v0.4/src/coolworld/ml/model.py:84: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
    self.spatial_encoder = nn.TransformerEncoder(layer, num_layers=spatial_layers)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
SKIPPED [1] mnt/data/SAM-WM-v0.4/tests/test_sam.py:26: could not import 'cvxpy': No module named 'cvxpy'
SKIPPED [1] mnt/data/SAM-WM-v0.4/tests/test_sam.py:48: could not import 'cvxpy': No module named 'cvxpy'
23 passed, 2 skipped, 1 deselected, 1 warning in 2.61s
```

`ruff` is declared in the development dependencies but is not installed in this artifact execution environment, so I am not claiming a Ruff pass here. Run `pip install -e ".[ml,research,dev]"` and `ruff check src tests scripts` locally before the Git commit.

Live MapLibre/OpenFreeMap/FortyGuard rendering is **not claimed as executed in this container** because it has no outbound DNS/API secret. No generated image or fake API response substitutes for that integration test. The real browser view must be verified from the user's VS Code environment.
