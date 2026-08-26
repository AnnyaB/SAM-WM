# Verification snapshot

Generated 2026-08-26.

Repository version: `1.0.1` environment/physics-integrity correction.

Local checks completed before packaging:
- `python -m compileall` on the package and entry points: PASS.
- `python -m pytest`: 12 invariant/unit tests PASS under the repository source path.
- Tooling contract: Python 3.12 only for the registered hackathon environment; CI, Makefile, README and Kaggle use `python -m ...` to prevent pyenv/venv interpreter mixing.
- synthetic end-to-end train -> validation calibration -> held-out evaluation: PASS.

Important scope boundary:
- The official Freiburg / Novi Sad / FAIRUrbTemp bytes were not available in this execution container, so real benchmark results are intentionally absent.
- The Freiburg and Novi Sad loaders use official dataset filenames/checksums and fail closed on schema mismatch.
- FAIRUrbTemp requires the official DOI 10.48620/93247 archive to be attached/extracted in Kaggle; no synthetic replacement is used.
- No FortyGuard API request was made during packaging and no API key is included.

Corrections in 1.0.1:
- CPython 3.12 is the registered interpreter across local development and CI.
- CI/Makefile/README use `python -m pytest` and `python -m ruff`, preventing pyenv/venv executable mixing.
- pytest has an explicit `src` path fallback for source-tree collection.
- SIGReg now regularizes trainable predicted latents; the detached JEPA target is not used for the regularizer.
- Conservative exchange is degree-normalized, preserving antisymmetry and a discrete maximum principle for the exchange step.
- The wind mechanism is masked out of the router when no observed wind is supplied.
- FortyGuard requests are intent-logged before POST; ambiguous POST outcomes fail closed; completed responses are content-addressed and reused without reposting.
