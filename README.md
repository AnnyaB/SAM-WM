# SAM-WM · CoolWorld
### Sparse Adaptive Mechanism World Model for evidence-bounded urban thermal intelligence

**Krsna · FortyGuard Hackathon'26 · Primary Track 1 — Resilient Cities & Infrastructure · Secondary Track 5 — Model Designing**

**Abstract.** CoolWorld turns real FortyGuard thermal evidence into short-horizon urban heat forecasts and intervention priorities using **SAM-WM**, a compact 117,705-parameter mechanism-structured world model. SAM-WM represents a city as a sparse physical graph, composes bounded local thermal mechanisms over a recurrent latent state, predicts +1…+6 h temperature fields, exposes calibrated uncertainty, and keeps forecasting separate from causal intervention claims. In a five-seed matched paper suite trained only on Freiburg, SAM-WM reaches **1.4515 ± 0.0149 °C MAE** on Freiburg held-out data and **1.4675 ± 0.0256 °C MAE** on zero-shot Novi Sad. The adapted iTransformer baseline degrades to **4.1367 ± 0.6737 °C** on Novi Sad and the adapted TimeMixer baseline to **2.7799 ± 0.7516 °C**. The result supports SAM-WM's cross-city transfer hypothesis, while ablations show that not every mechanism is equally supported by forecast MAE alone. The public product therefore uses SAM-WM as **decision support**, not as proof of causal cooling.

<p align="center">
  <b>[ <a href="https://sam-wm-coolworld.onrender.com">Live Demo</a> | <a href="results/paper_suite/paper_suite_results.json">Matched Results</a> | <a href="docs/RESEARCH_POSITIONING.md">Research Audit</a> | <a href="docs/KAGGLE_PAPER_SUITE.md">Reproduce</a> ]</b>
</p>

<p align="center">
  <img src="results/paper_suite/figures/main_horizon_results.svg" width="92%" alt="SAM-WM horizon-wise Freiburg and Novi Sad results">
</p>

> **Truth boundary.** Software does not physically cool a city. Trees, shade, reflective materials, water systems, retrofits, and other physical interventions do. CoolWorld forecasts where heat is likely to persist and helps prioritize where to investigate. Numerical cooling effects remain locked until independent treated/control evidence exists.

## What CoolWorld does

```text
REAL FORTYGUARD TEMPERATURE EVIDENCE
65 recorded hourly frames · 36-tile San José grid
                    │
                    ▼
               OBSERVE IN 3D
measured thermal field + provenance
                    │
                    ▼
                  SAM-WM
48 h context → sparse city graph → routed mechanisms → +1…+6 h rollout
                    │
                    ▼
            PRIORITIZE HOTSPOTS
future temperature + persistence + uncertainty
                    │
                    ▼
             ENGINEERING REVIEW
shade / canopy / reflective surface / other physical action
                    │
                    ▼
          MEASURE TREATED VS CONTROL
                    │
                    ▼
             VALIDATE OR ABSTAIN
```

The deployed UI follows **Observe → Forecast → Prioritize → Evidence**. It deliberately distinguishes:

- **measured evidence** from FortyGuard;
- **research forecasts** from SAM-WM;
- **operational certification** from the separate provider-replay gate;
- **causal cooling effects** from treated/control intervention evidence.

## SAM-WM

SAM-WM is a **Sparse Adaptive Mechanism World Model** for multi-step urban thermal fields. Its implemented hypothesis is that a small world model can transfer more reliably when the latent dynamics are coupled to local structure and bounded physical operators instead of relying only on unconstrained global sequence mixing.

At each rollout step it composes:

1. **Sparse adaptive mental map** — `O(E)` state-dependent message passing over a deterministic physical kNN graph.
2. **Conservative exchange** — symmetric learned conductance with antisymmetric pair heat flux; the exchange term sums to zero up to floating-point error.
3. **Wind transport** — conservative upwind transport when wind exists; exactly zero when wind is unavailable.
4. **Bounded source/sink forcing** — local unresolved forcing limited by a training-derived one-step bound.
5. **Bounded residual** — deliberately restricted free residual capacity.
6. **Adaptive routing** — state-dependent mixture of exchange, transport, source, and residual mechanisms.
7. **Recurrent latent dynamics** — autoregressive +1…+6 h state rollout.
8. **Uncertainty and surprise** — predictive scale plus source-validation split-conformal calibration.
9. **SIGReg** — adapted and attributed to LeWorldModel; SAM-WM does not claim authorship of SIGReg or copy LeWM's pixel architecture.

The full model is **physics-inspired / mechanism-structured**, not a complete first-principles urban energy-balance simulator. Exchange and transport are conservative operators; source and residual terms can add or remove local forcing, so the full model is not globally energy-conserving.

## Matched paper-suite protocol

The final matched experiment is machine-readable under [`results/paper_suite/`](results/paper_suite/): `paper_suite_results.json` is the index, `summary.json` contains all aggregate metrics, and `raw/*.json` contains every per-seed evaluation record.

- **Source city:** Freiburg.
- **Training:** Freiburg train split only.
- **Checkpoint selection:** Freiburg validation MAE only.
- **Calibration:** Freiburg validation residuals only.
- **ID test:** Freiburg held-out test.
- **OOD test:** Novi Sad, zero-shot.
- **Target adaptation:** none.
- **Target recalibration:** none.
- **Context / horizon:** 48 h → +1…+6 h.
- **Seeds:** `17, 29, 42, 73, 101`.
- **Models:** SAM-WM, iTransformer-adapted, TimeMixer-adapted, and five SAM-WM ablations.
- **FAIRUrbTemp/Turku:** not rerun in the matched deadline suite because the public FAIRUrbTemp host was unavailable. The earlier frozen v1 SAM-WM-only Turku result remains separate and is not mixed into the matched baseline table.

The two baseline implementations are **independent task adapters inspired by** iTransformer and TimeMixer. They are not the original authors' official implementations, so this repository does **not** claim to beat the official papers or universal SOTA.

## Results

### Freiburg held-out ID

| Model | MAE °C ↓ | RMSE °C ↓ | 90% conformal coverage | Params |
|---|---:|---:|---:|---:|
| **SAM-WM** | **1.4515 ± 0.0149** | 2.0483 ± 0.0072 | **90.45% ± 1.13%** | 117,705 |
| iTransformer-adapted | 1.6560 ± 0.0758 | 2.2879 ± 0.0997 | 87.96% ± 1.39% | 350,598 |
| TimeMixer-adapted | **1.4424 ± 0.0326** | **2.0023 ± 0.0484** | 89.49% ± 0.62% | 58,527 |

TimeMixer-adapted is slightly better on Freiburg source-domain MAE by **0.0092 °C (~0.64%)**. SAM-WM is not presented as the universal ID winner.

### Novi Sad zero-shot OOD

| Model | MAE °C ↓ | RMSE °C ↓ | 90% conformal coverage |
|---|---:|---:|---:|
| **SAM-WM** | **1.4675 ± 0.0256** | **2.1575 ± 0.0331** | **89.59% ± 0.99%** |
| iTransformer-adapted | 4.1367 ± 0.6737 | 5.1015 ± 0.7979 | 50.71% ± 7.42% |
| TimeMixer-adapted | 2.7799 ± 0.7516 | 3.4389 ± 0.7675 | 63.73% ± 17.47% |

Under this fixed source-only protocol, SAM-WM has **64.53% lower Novi Sad MAE than iTransformer-adapted** and **47.21% lower Novi Sad MAE than TimeMixer-adapted**. Its Freiburg→Novi Sad MAE increase is only **+0.0159 °C (+1.10%)**, versus **+2.4806 °C (+149.8%)** for iTransformer-adapted and **+1.3375 °C (+92.7%)** for TimeMixer-adapted.

<p align="center">
  <img src="results/paper_suite/figures/forecast_and_calibration.svg" width="92%" alt="Cross-city accuracy and source-frozen conformal coverage">
</p>

### What the baseline result means

The evidence supports a **cross-city transfer advantage under this matched protocol**. It does not prove that SAM-WM is globally superior to iTransformer or TimeMixer. The strongest defensible statement is:

> When all models are trained and selected only on Freiburg, SAM-WM preserves its error and calibration much better on zero-shot Novi Sad than the two matched adapted sequence-model baselines.

That is the central empirical result of the new paper suite.

## Ablations: which parts of SAM-WM are actually supported?

| Variant | Freiburg MAE ↓ | Novi Sad MAE ↓ | What the experiment says |
|---|---:|---:|---|
| **Full SAM-WM** | 1.4515 | 1.4675 | reference |
| − SIGReg | **1.3022** | 1.5669 | much better source fit, worse OOD + worse OOD coverage |
| − exchange | 1.4494 | 1.4715 | essentially tied in forecast MAE |
| − mental map | 1.4807 | 1.4872 | worse on both ID and OOD |
| − residual | 1.4572 | 1.5008 | slightly worse ID; clearly worse OOD |
| − RH | 1.5065 | **1.4434** | RH helps Freiburg, but Novi Sad has no RH; removing it slightly helps the missing-modality target |

<p align="center">
  <img src="results/paper_suite/figures/samwm_ablations.svg" width="92%" alt="SAM-WM ablation results">
</p>

The ablations are informative rather than uniformly flattering:

- **SIGReg shows a source-fit / transfer trade-off.** Removing it improves Freiburg MAE by ~10.3%, but worsens Novi Sad MAE by ~6.8% and drops Novi Sad coverage from ~89.6% to ~85.5%. This is consistent with the regularizer sacrificing some source fit for more transferable representation geometry.
- **Sparse mental-map updates are supported by the matched forecast metrics.** Removing them worsens both Freiburg and Novi Sad.
- **The bounded residual is supported, especially OOD.** Removing it increases Novi Sad MAE by ~2.27%.
- **Conservative exchange is not established as an accuracy win by this benchmark.** Removing it is nearly neutral in MAE. Its value is structural—antisymmetric pair exchange and an explicit conservation invariant—rather than a demonstrated large forecast-accuracy gain here.
- **RH exposes a real missing-modality issue.** RH improves Freiburg, but Novi Sad provides no RH. The no-RH ablation slightly outperforms the full model on Novi Sad. That result is retained, not hidden; future work should make modality dropout/invariance stronger.

## Efficiency

| Model | Trainable params | Freiburg latency / window* | Freiburg MAE |
|---|---:|---:|---:|
| SAM-WM | 117,705 | 0.525 ms | 1.4515 |
| iTransformer-adapted | 350,598 | **0.059 ms** | 1.6560 |
| TimeMixer-adapted | **58,527** | 0.761 ms | **1.4424** |

`*` Measured by the paper-suite evaluator on the Kaggle Tesla T4 runtime; these are practical implementation measurements, not hardware-independent microbenchmarks.

SAM-WM uses **66.4% fewer parameters than iTransformer-adapted** and is ~31% faster than the TimeMixer adapter in this evaluator, but iTransformer-adapted is much faster in raw GPU latency. The claim is therefore **balanced transfer/structure at compact scale**, not absolute latency leadership.

<p align="center">
  <img src="results/paper_suite/figures/efficiency.svg" width="92%" alt="Parameter and latency trade-offs">
</p>

All additional publication figures—including learning dynamics and representative forecast traces—live under [`results/paper_suite/figures/`](results/paper_suite/figures/).

## Real FortyGuard API evidence

FortyGuard is not decorative in this project. The deployed evidence bundle contains **65 compatible consecutive provider frames** on one **36-tile San José grid**. The request below is an actual tracked provider activity; the API key is intentionally absent.

Recorded request: [`artifacts/fortyguard/f5f0fdf137c5dedfef886e8dba32b5038b97f247640328110c67acbff8e096ee.activity.json`](artifacts/fortyguard/f5f0fdf137c5dedfef886e8dba32b5038b97f247640328110c67acbff8e096ee.activity.json)

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
    "features": [{
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-121.8955, 37.3295],
          [-121.8885, 37.3295],
          [-121.8885, 37.3345],
          [-121.8955, 37.3345],
          [-121.8955, 37.3295]
        ]]
      },
      "properties": {"name": "SAM-WM San Jose integration AOI"},
      "type": "Feature"
    }],
    "type": "FeatureCollection"
  }
}
```

Recorded response: [`artifacts/fortyguard/responses/5ba0d757a5ff2bd010f191fb1a1080a43cf7d408443554e3cb22ef25b1e7eca3.json`](artifacts/fortyguard/responses/5ba0d757a5ff2bd010f191fb1a1080a43cf7d408443554e3cb22ef25b1e7eca3.json)

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

The response is stored content-addressably; the activity record carries both request and response hashes. No provider key is committed anywhere in the repository.

## FortyGuard replay: research forecast ≠ operational certification

The frozen deployment checkpoint was also replayed on the recorded provider timeline without retraining:

| Item | Result |
|---|---:|
| Context | 48 h |
| Forecast horizon | 6 h |
| Provider windows | 12 |
| MAE | 2.047516 °C |
| RMSE | 2.501741 °C |
| Conformal radius | 3.211550 °C |
| Empirical coverage | **79.899691%** |
| Fixed minimum coverage | **80.000000%** |
| Operational certification | **FAIL** |

The miss is ~**0.1003 percentage points**. The threshold is not lowered after seeing the result. Therefore the product exposes the research forecast but labels the provider replay as **not operationally certified**. This fail-closed behavior is intentional technical evidence, not a hidden failure.

## Run from scratch

Python 3.11 or 3.12:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,app]'
make verify
python verify_runtime.py
make serve
```

Open:

```text
http://127.0.0.1:8000
```

The default demo uses recorded evidence and makes **zero new FortyGuard requests**.

### Docker

```bash
docker build -t sam-wm-coolworld .
docker run --rm -p 8000:7860 sam-wm-coolworld
```

Then open `http://127.0.0.1:8000`.

## Reproduce the matched paper suite

The exact fixed protocol is in [`config/paper.yaml`](config/paper.yaml). The deadline artifact was run as:

```bash
python paper_suite.py \
  --config config/paper.yaml \
  --mode paper \
  --skip-fairurb \
  --out artifacts/paper_suite
```

This runs five seeds for the full model, two matched adapted baselines, and five ablations. Completed checkpoints resume automatically. Figures are generated from machine-readable results; values are not hand-entered into plots.

The full Kaggle archive additionally contains the 40 checkpoints and histories. Binary checkpoints are intentionally not committed to this Git repository; the tracked [`results/paper_suite/`](results/paper_suite/) directory contains the complete aggregate summary, every per-seed raw evaluation JSON, manifest, and all six publication-figure families as editable SVGs. The original Kaggle archive retains the vector PDFs, 600-dpi PNGs, and all 40 binary checkpoints.

## What does not work yet

- **Operational provider certification is still closed.** The frozen FortyGuard replay misses the preregistered 80% coverage gate by ~0.1003 percentage points.
- **No causal cooling-effect estimate is claimed.** There is not yet an independent treated/control intervention trial proving that a suggested physical action caused a measured temperature reduction.
- **The matched baseline suite currently has two domains, not three.** FAIRUrbTemp/Turku could not be rerun during the deadline window because its public host was unavailable. The earlier frozen v1 SAM-WM-only Turku result is not used to inflate the matched comparison.
- **The baseline adapters are not official iTransformer/TimeMixer reproductions.** They are fixed independent task adapters inspired by those architectures.
- **Exchange is structurally justified but not an accuracy win in this ablation.** More targeted tests are needed to establish when its inductive bias helps prediction.
- **Missing-modality robustness can improve.** The no-RH ablation slightly improves Novi Sad, where RH is unavailable.
- **The public deployment keeps live provider calls disabled by default.** Live mode requires a server-side key plus an explicit enable flag.

## Repository map

```text
SAM-WM/
├── src/coolworld/samwm.py          # core SAM-WM mechanisms
├── src/coolworld/paper_models.py   # matched baselines + ablations
├── src/coolworld/paper_suite.py    # five-seed paper protocol
├── src/coolworld/paper_figures.py  # publication figures
├── paper_suite.py                  # benchmark entry point
├── config/paper.yaml               # fixed matched protocol
├── results/paper_suite/            # tracked final result JSON + SVG figures
├── artifacts/                      # frozen runtime/provider evidence
├── static/                         # 3D CoolWorld interface
├── tests/                          # unit/runtime contracts
├── docs/                           # protocol, production, research audit
├── Dockerfile
├── Makefile
└── README.md
```

## Hackathon scope and reproducibility

This repository was created on **18 August 2026**, after the Hackathon'26 kickoff. The submission-specific CoolWorld integration, provider evidence pipeline, deployment, evaluation contracts, and paper-suite additions are tracked in Git history.

The live deployment is designed to be judge-safe:

- no login required;
- no browser-exposed provider key;
- recorded evidence by default;
- immutable checkpoint/evidence hashes;
- fail-closed operational and causal gates;
- `/api/health` for runtime checks;
- deterministic provider evidence replay where applicable.

## AI assistance disclosure

**OpenAI ChatGPT** was used as a development assistant for code review, debugging, experiment orchestration, scientific-result auditing, and documentation. AI assistance did **not** generate the recorded FortyGuard measurements or fabricate benchmark metrics; reported numbers come from tracked machine-readable artifacts. The deployed CoolWorld product does not depend on a runtime LLM for its thermal forecasts or truth-state logic.

## Attribution

- **SIGReg:** adapted/attributed to Maes et al., *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels* (2026).
- **iTransformer-adapted:** independent task adapter inspired by Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024 Spotlight. The official implementation is not bundled here.
- **TimeMixer-adapted:** independent task adapter inspired by Wang et al., *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*, ICLR 2024. The official implementation is not bundled here.

SAM-WM's mechanism composition, sparse thermal graph formulation, bounded source/residual design, provider evidence gates, and CoolWorld integration are the contribution tested in this repository.

## License

MIT — see [`LICENSE`](LICENSE).
