# Verification status — v0.1

This file records what was actually executed while creating this build.

## Verified here

- Python source compilation: **PASS**.
- Non-live tests: **8 passed**.
- Evidence policy contains no simulator/synthetic evidence class.
- FortyGuard provenance requires activity ID + request hash.
- Content-addressed evidence rejects mismatched hashes.
- FortyGuard heatmap schema rejects undocumented granularity values.
- Hackathon date floor rejects dates before 2021-01-01.
- SAM support-deficiency spectral/symmetry checks pass.
- SAM orthogonal-equivariance check passes.

## Not executed here

- **Live FortyGuard request**: not executed because this artifact runtime has no outbound network/API-secret injection. No cached or fabricated response was substituted.
- **Live Phoenix ArcGIS integration test**: intentionally not executed for the same network reason.
- **Exact CVXPY QCQP tests**: source is present and `cvxpy` is a declared project dependency, but CVXPY is not installed in this artifact execution environment. Two exact-solver tests therefore reported `SKIPPED`; they did not report false passes.

Before any empirical result or paper claim, run the live integration suite and exact-solver tests in the real project environment.

## No result claims

This v0.1 repository does **not** claim a trained world model, measured cooling benefit, causal intervention effect, hackathon score, paper result, or lives saved. Those claims require real evidence that has not yet been collected through this build.

## v0.2 verification

- Non-live tests after 3D/world-mode/model-gate changes: **14 passed**.
- Exact CVXPY reference tests: **2 skipped** because CVXPY is not installed in this artifact execution environment; they are not reported as passes.
- Live network integration test: **deselected**, so no external response was faked.
- Python source compilation: **PASS**.
- API secret scan: no literal FortyGuard API key value is stored in source/config/static files.
- Counterfactual endpoint fails closed until an evidence-bearing model artifact is present.
- Validated replay has no generated fallback and reports `REPLAY_NOT_AVAILABLE` until real before/after/control evidence exists.

No claim of measured cooling, causal effect, global validation, lives saved, or trained world-model performance is made by v0.2.
