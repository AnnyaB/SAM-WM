# Safe migration and continuation state

This document is the canonical hand-off for the 26 Aug 2026 repository rescue. It exists so a later chat/session can continue without guessing.

## Repository boundary

Only the private GitHub repository `AnnyaB/SAM-WM` and its branch `sam-wm-v1-mechanism-redesign` are in scope. LAADAN-AC, the ESP32 repository, portfolio, FYP files, and every other local/GitHub project are out of scope and must not be modified.

The safe local clone is expected at:

```text
~/Downloads/SAM-WM-SAFE
```

Before any destructive copy or Git write, verify all three invariants:

```text
git root == ~/Downloads/SAM-WM-SAFE
origin   == https://github.com/AnnyaB/SAM-WM.git
branch   == sam-wm-v1-mechanism-redesign
```

## Secret policy

The real FortyGuard API key is local-only. `.env` is ignored and must never be copied into the research ZIP, printed, staged, or committed. Migration commands must exclude `.env` explicitly.

## Current correction

A previous local verification mixed interpreters: the editable package was installed into a Python 3.14 venv while the shell-resolved `pytest` came from pyenv Python 3.12.6. The registered environment is now **CPython 3.12**, and all tools are invoked through the same interpreter as `python -m ...`.

Repository version `1.0.1` also corrects three core integrity issues discovered during audit:

1. SIGReg now acts on trainable predicted latents rather than the detached JEPA target.
2. Conservative exchange is degree-normalized, retaining exact antisymmetry while enforcing a one-step discrete maximum principle.
3. When observed wind is unavailable, the router masks the wind-transport mechanism instead of allocating probability to an unsupported branch.

The FortyGuard client is also fail-closed around ambiguous POST outcomes and content-addresses completed responses for crash-safe reuse.

## Verification before real benchmarks

The packaged source has passed:

```text
python -m compileall   PASS
python -m pytest       12/12 PASS
synthetic train/eval   PASS
```

These are software tests, not research results. Freiburg, Novi Sad, FAIRUrbTemp and live FortyGuard evidence must still be run against their real source bytes before any result claim.

## Next execution order

1. Replace only `~/Downloads/SAM-WM-SAFE` with the corrected bundle, excluding `.git`, `.env`, and `.venv`.
2. Recreate `.venv` with `~/.pyenv/versions/3.12.6/bin/python` (or another verified CPython 3.12 executable).
3. Run `./.venv/bin/python -m pip install -e '.[dev]'`.
4. Run `./.venv/bin/python -m compileall ...` and `./.venv/bin/python -m pytest`; expected test count: 12.
5. Inspect `git status`; it must mention SAM-WM files only.
6. Commit/push only `sam-wm-v1-mechanism-redesign` after verification.
7. Then run the Kaggle notebook from the clean branch: real Freiburg preflight -> 3-seed training -> Freiburg held-out evaluation -> zero-shot Novi Sad -> zero-shot FAIRUrbTemp -> results/figures.
8. Only after real results exist, wire the trained checkpoint and CANDRA evidence gate into the separately reviewed 3D UI.

No SOTA, AGI/human-child equivalence, causal intervention cooling, or planetary cooling claim is licensed before the corresponding experiments/evidence exist.
