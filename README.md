# SAM-WM · CoolWorld

**Sparse Adaptive Mechanism World Model for urban thermal forecasting**

**Krsna · FortyGuard Hackathon'26 · Track 1 — Resilient Cities & Infrastructure**

CoolWorld uses real FortyGuard street-level thermal evidence to forecast short-horizon urban heat and rank future persistent hotspots for engineering review. The model is **SAM-WM**, a compact 117,705-parameter world model with a sparse physical graph, typed thermal mechanisms, recurrent rollout, and calibrated uncertainty.

<p align="center">
  <b><a href="https://sam-wm-coolworld.onrender.com/">Live demo</a> · <a href="results/paper_suite/paper_suite_results.json">Exact benchmark results</a> · <a href="docs/KAGGLE_PAPER_SUITE.md">Reproduction protocol</a></b>
</p>

> **Claim boundary.** SAM-WM forecasts where heat is likely to persist. It does not claim that a proposed tree, shade structure, reflective material, or other intervention caused cooling unless independent treated/control evidence exists.

## Method

For a city graph $G=(V,E)$, node temperature $T_i^t$, and latent state $z_i^t$, SAM-WM advances the thermal field as

$$
T_i^{t+1}
=
T_i^t
+
\Delta_i^{\mathrm{ex}}
+
\Delta_i^{\mathrm{wind}}
+
\Delta_i^{\mathrm{src}}
+
\Delta_i^{\mathrm{res}}.
$$

The four operators are routed explicitly rather than collapsed into one unrestricted update:

- **Sparse adaptive mental map:** state-dependent message passing over a deterministic physical kNN graph.
- **Conservative exchange:** non-negative symmetric conductance with antisymmetric pair flux $F_{ij}=k_{ij}(T_j-T_i)$.
- **Wind transport:** conservative upwind transport when observed wind is available; exactly disabled when unavailable.
- **Bounded source/sink:** learned local forcing with a source-domain one-step bound.
- **Bounded residual:** limited free correction capacity rather than an unconstrained bypass.
- **Adaptive routing and recurrent rollout:** mechanism weights depend on latent state and time; the latent state is advanced recurrently for +1…+6 h.
- **Uncertainty:** a learned Laplace scale is calibrated from Freiburg validation residuals and then frozen for OOD evaluation.

The implementation matches these semantics directly in [`src/coolworld/samwm.py`](src/coolworld/samwm.py). The physical graph is built from local city-centred geometry in [`src/coolworld/graph.py`](src/coolworld/graph.py).

SIGReg is **not** claimed as an original SAM-WM contribution. It is an attributed adaptation of the LeJEPA/LeWM regularization line. SAM-WM's original hypothesis is the sparse, mechanism-structured continuous-field dynamics together with an evidence-bounded deployment contract.

## Final matched benchmark

The final Kaggle paper suite completed **40/40 fits**: 8 model/ablation families × 5 seeds (`17, 29, 42, 73, 101`).

Protocol:

- train: Freiburg train split only;
- checkpoint selection: Freiburg validation MAE only;
- uncertainty calibration: Freiburg validation residuals only;
- ID test: Freiburg held-out;
- OOD test: Novi Sad, zero-shot;
- target fine-tuning: none;
- target recalibration: none;
- context / horizon: **48 h → +1…+6 h**.

The iTransformer-adapted and TimeMixer-adapted baselines are matched task adapters inspired by the peer-reviewed architectures; they are not presented as reproductions of the authors' official code.

### Accuracy and frozen-calibration transfer

<p align="center">
  <img src="results/paper_suite/figures/forecast_and_calibration.svg" width="94%" alt="Final Kaggle cross-city forecast accuracy and frozen source calibration">
</p>

| Model | Freiburg MAE °C ↓ | Freiburg RMSE °C ↓ | Freiburg coverage | Novi Sad MAE °C ↓ | Novi Sad RMSE °C ↓ | Novi Sad coverage |
|---|---:|---:|---:|---:|---:|---:|
| **SAM-WM** | 1.4515 ± 0.0149 | 2.0483 ± 0.0072 | **90.45% ± 1.13%** | **1.4675 ± 0.0256** | **2.1575 ± 0.0331** | **89.59% ± 0.99%** |
| iTransformer-adapted | 1.6560 ± 0.0758 | 2.2879 ± 0.0997 | 87.96% ± 1.39% | 4.1367 ± 0.6737 | 5.1015 ± 0.7979 | 50.71% ± 7.42% |
| TimeMixer-adapted | **1.4424 ± 0.0326** | **2.0023 ± 0.0484** | 89.49% ± 0.62% | 2.7799 ± 0.7516 | 3.4389 ± 0.7675 | 63.73% ± 17.47% |

TimeMixer-adapted is marginally better on Freiburg MAE by about **0.0092 °C (0.64%)**. Under Freiburg→Novi Sad transfer, MAE changes by **+1.10%** for SAM-WM, **+149.8%** for iTransformer-adapted, and **+92.7%** for TimeMixer-adapted. On Novi Sad, SAM-WM has **64.53% lower MAE** than the matched iTransformer adapter and **47.21% lower MAE** than the matched TimeMixer adapter.

### Error across the +1…+6 h rollout

<p align="center">
  <img src="results/paper_suite/figures/main_horizon_results.svg" width="94%" alt="Final Kaggle horizon-wise Freiburg ID and Novi Sad OOD results">
</p>

The bands are the actual five-seed standard deviations stored in the final result object.

### Ablations

<p align="center">
  <img src="results/paper_suite/figures/samwm_ablations.svg" width="94%" alt="Final Kaggle SAM-WM ablation study">
</p>

| Variant | Freiburg MAE ↓ | Novi Sad MAE ↓ | Supported interpretation |
|---|---:|---:|---|
| **Full SAM-WM** | 1.4515 | 1.4675 | reference |
| − SIGReg | **1.3022** | 1.5669 | stronger source fit, weaker transfer/calibration |
| − exchange | 1.4494 | 1.4715 | aggregate MAE essentially tied |
| − mental map | 1.4807 | 1.4872 | worse ID and OOD |
| − residual | 1.4572 | 1.5008 | small ID loss, larger OOD loss |
| − RH | 1.5065 | **1.4434** | source RH helps Freiburg; missing target RH creates modality shift |

The ablation results are reported as observed. In particular, the conservative exchange operator is structurally verified but does not show a material aggregate-MAE gain in this benchmark.

### Representative zero-shot rollout

<p align="center">
  <img src="results/paper_suite/figures/forecast_trace.svg" width="94%" alt="Representative Novi Sad zero-shot forecast trace">
</p>

`efficiency.svg` and `learning_curves.svg` are retained under [`results/paper_suite/figures/`](results/paper_suite/figures/) as supplementary diagnostics.

**Figure provenance.** Five tracked SVGs are direct exports from the completed Kaggle archive. `forecast_trace.svg` is a layout-only re-render from the exact saved final trace arrays so that the legend no longer overlaps the data. No observed value, prediction, model, split, seed, or metric was changed.

## Real FortyGuard integration

The Temperature API is part of the evidence path, not a decorative data source. The repository contains **65 recorded hourly TCM frames** over one **36-tile San José, California grid**, with content-addressed request/response evidence. The public deployment replays the recorded evidence by default and keeps the provider key server-side.

### One real request

Full tracked request: [`artifacts/fortyguard/f5f0fdf137c5dedfef886e8dba32b5038b97f247640328110c67acbff8e096ee.activity.json`](artifacts/fortyguard/f5f0fdf137c5dedfef886e8dba32b5038b97f247640328110c67acbff8e096ee.activity.json)

```json
{
  "analytic_type": "tcm",
  "date_time": {
    "filter_type": 1,
    "start_date": "2026-08-21",
    "start_time": "14:00"
  },
  "granularity": 100,
  "polygon_aoi": {
    "type": "FeatureCollection",
    "features": [{
      "properties": {"name": "SAM-WM San Jose integration AOI"},
      "geometry": {"type": "Polygon", "coordinates": "see tracked request"}
    }]
  }
}
```

Request SHA-256:

```text
f5f0fdf137c5dedfef886e8dba32b5038b97f247640328110c67acbff8e096ee
```

### Corresponding real response

Full tracked response: [`artifacts/fortyguard/responses/5ba0d757a5ff2bd010f191fb1a1080a43cf7d408443554e3cb22ef25b1e7eca3.json`](artifacts/fortyguard/responses/5ba0d757a5ff2bd010f191fb1a1080a43cf7d408443554e3cb22ef25b1e7eca3.json)

```json
{
  "data": {
    "activity_id": "485b7652-5225-4944-a822-cd8189af4d91",
    "result": {
      "map_data": {
        "features": [{
          "id": "0",
          "properties": {
            "average_temperature": 24.3268,
            "max_temperature": 24.3268,
            "min_temperature": 24.3268,
            "tile_id": 0
          }
        }]
      }
    }
  }
}
```

Response SHA-256:

```text
5ba0d757a5ff2bd010f191fb1a1080a43cf7d408443554e3cb22ef25b1e7eca3
```

The custom provider client records request intent before POST, stores the returned activity ID, resumes polling without silently repeating an ambiguous request, hashes completed responses, and fails closed on evidence mismatch. No FortyGuard API key is committed.

## Resilient-city workflow

CoolWorld is intended for a city resilience / urban-infrastructure team deciding where limited heat-mitigation review should go first:

```text
real FortyGuard thermal evidence
        ↓
observe measured field + timestamp + provenance
        ↓
48 h SAM-WM context
        ↓
sparse graph → routed mechanisms → recurrent +1…+6 h rollout
        ↓
future temperature + persistence + uncertainty
        ↓
rank hotspots for engineering review
        ↓
choose a physical intervention outside the model
        ↓
measure treated vs control
        ↓
validate effect or abstain
```

The software deliberately separates **observed evidence**, **research forecast**, **operational certification**, and **causal intervention evidence**.

### Provider replay gate

The promoted deployment checkpoint was replayed, without retraining, on the recorded FortyGuard timeline.

| Item | Result |
|---|---:|
| Context | 48 h |
| Forecast horizon | 6 h |
| Provider windows | 12 |
| MAE | 2.047516 °C |
| RMSE | 2.501741 °C |
| Conformal radius | 3.211550 °C |
| Empirical coverage | **79.899691%** |
| Fixed minimum | **80.000000%** |
| Operational certification | **FAIL** |

The threshold is not lowered after evaluation. The application exposes the frozen model only as a clearly labelled **research forecast** while the fixed operational gate remains failed.

## Run from scratch

Supported Python: **3.11 or 3.12**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,app]'
make verify
python verify_runtime.py
make serve
```

Open `http://127.0.0.1:8000`.

Health check:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Docker:

```bash
docker build -t sam-wm-coolworld .
docker run --rm -p 7860:7860 \
  -e COOLWORLD_LIVE_API_ENABLED=0 \
  sam-wm-coolworld
```

Then open `http://127.0.0.1:7860`.

## Training and reproduction

Single SAM-WM training run:

```bash
python train.py --seed 17
```

The five-seed matched paper-suite protocol is documented in [`docs/KAGGLE_PAPER_SUITE.md`](docs/KAGGLE_PAPER_SUITE.md). Publication plots are generated by [`scripts/plot_results.py`](scripts/plot_results.py). The judge-readable final export is under [`results/paper_suite/`](results/paper_suite/).

The repository retains the selected final paper-suite checkpoint at [`results/paper_suite/checkpoints/samwm_seed42_best.pt`](results/paper_suite/checkpoints/samwm_seed42_best.pt). Seed 42 was selected by Freiburg validation MAE only (`1.3881008397 °C`); its SHA-256 is:

```text
d29e2939f86e7d6961dd16b6d2e5e20a2868d1c003825ef5c0ad2eae996f18dc
```

The live application intentionally uses the separate hash-locked deployment bundle under `artifacts/deployment/`. Replacing it with the paper-suite checkpoint without regenerating its calibration/evaluation/replay chain would invalidate the deployment evidence.

## What does not work yet

- The provider replay misses the fixed 80% coverage gate by ~0.1003 percentage points; the deployment is **not operationally certified**.
- No independent treated/control intervention-effect artifact exists, so numerical causal cooling effects are unavailable.
- The matched external baselines are task adapters, not official-code reproductions.
- The final deadline paper suite has one OOD city, Novi Sad. Turku was deferred when the FAIRUrbTemp host was unavailable during the final run.
- Conservative exchange is structurally verified but has not shown a material aggregate-MAE advantage in this benchmark.
- Missing-modality transfer remains a limitation; the Novi Sad `−RH` ablation exposes source/target modality mismatch.
- No result establishes a causal temperature reduction from a future urban intervention.

## Development provenance and disclosure

This repository's submitted project work was developed during the hackathon build window. The project does **not** claim to have started from the FortyGuard Temperature API Quickstart; the provider integration in [`src/coolworld/fortyguard.py`](src/coolworld/fortyguard.py) and [`fortyguard_collect.py`](fortyguard_collect.py) is a custom implementation. If any Quickstart-derived code is later introduced, it must be disclosed here before submission.

AI tools were used during development for code review, debugging, test generation, documentation editing, and figure-layout review. Training outputs, model evaluations, provider responses, thresholds, and benchmark figures are grounded in repository/Kaggle artifacts rather than generated by an assistant.

## References

- Amini, S., Huerta, A., Franke, J. *et al.* (2026). *Comprehensive compilation and quality assessment of street-level urban air temperature measurements across European networks*. Scientific Data 13, 658. https://doi.org/10.1038/s41597-026-06804-4
- Baek, D., Lee, G., Baek, J., Lee, H. and Ahn, S. (2026). *Learning to Theorize the World from Observation*. ICML 2026, Oral.
- Baek, J., Wu, Y.-F., Singh, G. and Ahn, S. (2025). *Dreamweaver: Learning Compositional World Models from Pixels*. ICLR 2025.
- Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L. and Long, M. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR 2024.
- Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J.Y. and Zhou, J. (2024). *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*. ICLR 2024.
- Balestriero, R. and LeCun, Y. (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*. arXiv:2511.08544. SIGReg originates in this line.
- Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y. and Balestriero, R. (2026). *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*. arXiv:2603.19312.

## License

See [`LICENSE`](LICENSE).
