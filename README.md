# SAM-WM · CoolWorld
### Sparse Adaptive Mechanism World Model for urban thermal forecasting

**Krsna · FortyGuard Hackathon 2026 · Track 1 — Resilient Cities & Infrastructure**

**Abstract.** CoolWorld turns real FortyGuard street-level thermal evidence into short-horizon urban heat forecasts and persistent-hotspot priorities using **SAM-WM**, a compact 117,705-parameter world model. SAM-WM represents a city as a sparse physical graph and rolls temperature forward through routed exchange, transport, source/sink and bounded residual mechanisms while carrying forecast uncertainty. In the final five-seed matched suite, trained, selected and calibrated on Freiburg only, SAM-WM reaches **1.4515 ± 0.0149 °C MAE** on Freiburg held-out data and **1.4675 ± 0.0256 °C MAE** on zero-shot Novi Sad. Under the same protocol, the matched iTransformer-adapted and TimeMixer-adapted baselines reach **4.1367 ± 0.6737 °C** and **2.7799 ± 0.7516 °C** on Novi Sad. The strongest supported result is **cross-city preservation**, not universal SOTA and not a causal cooling claim.

<p align="center">
  <b>[ <a href="https://sam-wm-coolworld.onrender.com/">Live Demo</a> · <a href="results/paper_suite/paper_suite_results.json">Exact Results</a> · <a href="docs/KAGGLE_PAPER_SUITE.md">Reproduce</a> · <a href="docs/RESEARCH_POSITIONING.md">Research Audit</a> ]</b>
</p>

> **Claim boundary.** CoolWorld does not physically cool a city. Shade, trees, reflective materials, water systems and other physical interventions can change thermal conditions. SAM-WM forecasts where heat is likely to persist so an engineer can prioritize review. A numerical intervention effect remains unavailable until independent treated/control evidence supports it.

## What SAM-WM does

For a city graph \(G=(V,E)\), node temperature \(T_i^t\) and latent state \(z_i^t\), SAM-WM advances the field as

\[
T_i^{t+1}=T_i^t+\Delta_i^{\mathrm{ex}}+\Delta_i^{\mathrm{wind}}+\Delta_i^{\mathrm{src}}+\Delta_i^{\mathrm{res}}.
\]

The terms are deliberately typed rather than collapsed into one unconstrained update:

- **Sparse adaptive mental map** — state-dependent message passing over a deterministic physical kNN graph.
- **Conservative exchange** — non-negative symmetric conductance with antisymmetric pair flux \(F_{ij}=k_{ij}(T_j-T_i)\); the exchange term alone sums to approximately zero up to floating-point error.
- **Wind transport** — conservative upwind transport when observed wind is available; disabled when it is unavailable.
- **Bounded source/sink** — learned local forcing limited by a one-step bound derived from source-domain observations.
- **Bounded residual** — restricted free correction capacity rather than an unconstrained bypass.
- **Adaptive routing + recurrent rollout** — a learned router changes mechanism mixture with state/time while a recurrent latent state advances the +1…+6 h forecast.
- **Uncertainty** — a learned scale is calibrated from Freiburg validation residuals and then frozen for OOD evaluation.

SIGReg is **not** an original SAM-WM contribution. It is an attributed adaptation of the LeJEPA/LeWM regularization line; SAM-WM's original hypothesis is the sparse, mechanism-structured continuous-field dynamics and evidence-bounded deployment design.

## Final matched benchmark

The authoritative final Kaggle archive completed **40/40 fits**: 8 model/ablation families × 5 seeds (`17, 29, 42, 73, 101`).

- train: Freiburg train split only;
- select checkpoint: Freiburg validation MAE only;
- calibrate uncertainty: Freiburg validation residuals only;
- ID test: Freiburg held-out;
- OOD test: Novi Sad, zero-shot;
- target fine-tuning: **none**;
- target recalibration: **none**;
- context / horizon: **48 h → +1…+6 h**.

The external baselines are matched independent adapters inspired by the peer-reviewed iTransformer and TimeMixer architectures. They are not the authors' official implementations, so this is a controlled internal comparison rather than an official-paper or universal-SOTA claim.

### Accuracy and uncertainty transfer

<p align="center">
  <img src="results/paper_suite/figures/forecast_and_calibration.svg" width="94%" alt="Final Kaggle cross-city forecast accuracy and frozen source calibration">
</p>

All values are five-seed **mean ± SD** from the exact final `paper_suite_results.json`.

| Model | Freiburg MAE °C ↓ | Freiburg RMSE °C ↓ | Freiburg coverage | Novi Sad MAE °C ↓ | Novi Sad RMSE °C ↓ | Novi Sad coverage |
|---|---:|---:|---:|---:|---:|---:|
| **SAM-WM** | 1.4515 ± 0.0149 | 2.0483 ± 0.0072 | **90.45% ± 1.13%** | **1.4675 ± 0.0256** | **2.1575 ± 0.0331** | **89.59% ± 0.99%** |
| iTransformer-adapted | 1.6560 ± 0.0758 | 2.2879 ± 0.0997 | 87.96% ± 1.39% | 4.1367 ± 0.6737 | 5.1015 ± 0.7979 | 50.71% ± 7.42% |
| TimeMixer-adapted | **1.4424 ± 0.0326** | **2.0023 ± 0.0484** | 89.49% ± 0.62% | 2.7799 ± 0.7516 | 3.4389 ± 0.7675 | 63.73% ± 17.47% |

TimeMixer-adapted is marginally better on Freiburg MAE by about **0.0092 °C (0.64%)**. The meaningful difference appears under transfer: Freiburg→Novi Sad MAE changes by **+1.10%** for SAM-WM, **+149.8%** for iTransformer-adapted and **+92.7%** for TimeMixer-adapted. On Novi Sad, SAM-WM has **64.53% lower MAE** than the matched iTransformer adapter and **47.21% lower MAE** than the matched TimeMixer adapter.

### Error across the +1…+6 h rollout

<p align="center">
  <img src="results/paper_suite/figures/main_horizon_results.svg" width="94%" alt="Final Kaggle horizon-wise Freiburg ID and Novi Sad OOD results">
</p>

The bands are the actual five-seed standard deviations from the final result object.

### Ablations: what the experiment supports

<p align="center">
  <img src="results/paper_suite/figures/samwm_ablations.svg" width="94%" alt="Final Kaggle SAM-WM ablation study">
</p>

| Variant | Freiburg MAE ↓ | Novi Sad MAE ↓ | Evidence from this benchmark |
|---|---:|---:|---|
| **Full SAM-WM** | 1.4515 | 1.4675 | reference |
| − SIGReg | **1.3022** | 1.5669 | stronger source fit, weaker transfer and OOD coverage |
| − exchange | 1.4494 | 1.4715 | aggregate MAE essentially tied |
| − mental map | 1.4807 | 1.4872 | worse ID and OOD |
| − residual | 1.4572 | 1.5008 | small ID loss, larger OOD loss |
| − RH | 1.5065 | **1.4434** | source RH helps Freiburg; Novi Sad's missing RH exposes modality shift |

The ablation evidence is intentionally not made artificially uniform. The sparse mental map helps aggregate ID and OOD error; bounded residual capacity matters more OOD; SIGReg trades source fit for transfer/calibration. The conservative exchange operator is structurally verified but does **not** yet show a material aggregate-MAE gain. The `−RH` result exposes a real missing-modality weakness to address in future work.

### Representative zero-shot rollout

<p align="center">
  <img src="results/paper_suite/figures/forecast_trace.svg" width="94%" alt="Final Kaggle representative Novi Sad zero-shot forecast trace">
</p>

The remaining final figures, including learning dynamics and efficiency, are preserved under [`results/paper_suite/figures/`](results/paper_suite/figures/). They are copied directly from the completed Kaggle archive.

## Real FortyGuard integration

FortyGuard is central to the deployed CoolWorld workflow, not a decorative layer. The repository contains **65 compatible recorded hourly TCM frames** over one **36-tile San José, California grid**, with content-addressed request/response evidence. The public deployment replays that immutable real evidence by default; it does not expose an API key client-side.

### One real request

Tracked request: [`artifacts/fortyguard/f5f0fdf137c5dedfef886e8dba32b5038b97f247640328110c67acbff8e096ee.activity.json`](artifacts/fortyguard/f5f0fdf137c5dedfef886e8dba32b5038b97f247640328110c67acbff8e096ee.activity.json)

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

The completed provider activity is `485b7652-5225-4944-a822-cd8189af4d91`. Request SHA-256:

```text
f5f0fdf137c5dedfef886e8dba32b5038b97f247640328110c67acbff8e096ee
```

### Corresponding real response excerpt

Tracked response: [`artifacts/fortyguard/responses/5ba0d757a5ff2bd010f191fb1a1080a43cf7d408443554e3cb22ef25b1e7eca3.json`](artifacts/fortyguard/responses/5ba0d757a5ff2bd010f191fb1a1080a43cf7d408443554e3cb22ef25b1e7eca3.json)

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

The custom client persists request intent before POST, stores the returned activity ID, resumes polling without silently re-spending an ambiguous request, hashes the completed response, and fails closed on evidence mismatch. No FortyGuard API key is committed.

## From forecast to a real resilient-city decision

CoolWorld is designed for a resilience/urban-infrastructure team deciding **where scarce field-review and heat-mitigation effort should go first**:

`real thermal evidence → observe in 3D → 48 h SAM-WM context → +1…+6 h forecast → persistent-hotspot ranking → engineering review → physical intervention → treated/control measurement → validate or abstain`

The application deliberately separates four states: **observed evidence**, **research forecast**, **operational certification**, and **causal intervention evidence**. A useful forecast does not automatically become a certified operational decision, and neither state proves the effect of a proposed tree, shade structure or reflective material.

### Provider replay gate

The promoted deployment checkpoint was replayed, without retraining, on the recorded FortyGuard timeline:

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

That miss is kept. The threshold is not lowered after evaluation. The live product can expose a clearly labelled **research forecast** while continuing to report that the fixed operational certification gate did not pass.

## Using the code

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

Single SAM-WM training:

```bash
python train.py --seed 17
```

Final five-seed matched suite used the protocol documented in [`docs/KAGGLE_PAPER_SUITE.md`](docs/KAGGLE_PAPER_SUITE.md). The plotting entry point used by the final Kaggle assembly is:

```bash
python scripts/plot_results.py \
  --results artifacts/paper_suite/paper_suite_results.json \
  --root artifacts/paper_suite \
  --out artifacts/paper_suite/figures
```

The tracked judge-readable export is under [`results/paper_suite/`](results/paper_suite/). The original Kaggle ZIP contains all 40 checkpoints, all 40 histories and PDF/SVG/600-dpi PNG figures. Git keeps the exact final result JSON, the six exact SVG figures and the **selected final SAM-WM paper-suite checkpoint** at [`results/paper_suite/checkpoints/samwm_seed42_best.pt`](results/paper_suite/checkpoints/samwm_seed42_best.pt).

### Paper-suite checkpoint vs live deployment checkpoint

The final paper suite selects **seed 42** by Freiburg validation MAE only (`1.3881008397 °C`), with checkpoint SHA-256 `d29e2939f86e7d6961dd16b6d2e5e20a2868d1c003825ef5c0ad2eae996f18dc`. See [`SELECTED_CHECKPOINT.json`](results/paper_suite/SELECTED_CHECKPOINT.json).

The live app intentionally continues to use the already promoted, hash-locked bundle under `artifacts/deployment/`. Replacing that file with a research checkpoint without regenerating its calibration/evaluation/replay chain would invalidate the deployment evidence, so the two artifacts are kept separate.

## What does not work yet?

- The provider replay misses the fixed 80% coverage gate by ~0.1003 percentage points, so the deployment is **not operationally certified**.
- CANDRA has no independent treated/control intervention-effect artifact; quantitative causal cooling effects remain unavailable.
- The matched external baselines are architecture-faithful task adapters, not official-code reproductions.
- The final matched deadline suite has one OOD city, Novi Sad. **Turku was deferred** because the FAIRUrbTemp host was unavailable during the final run.
- The conservative exchange operator is structurally verified but has not shown a material MAE advantage in this benchmark.
- Missing-modality transfer needs improvement; the Novi Sad `−RH` ablation exposes source/target modality mismatch.
- No result here establishes a causal temperature reduction from a future urban intervention.

## Development provenance and AI disclosure

This repository was initialized independently **after the 18 August 2026 kickoff** and was **not cloned from the FortyGuard Temperature API Quickstart**. The Temperature API integration is implemented in the custom fail-closed client [`src/coolworld/fortyguard.py`](src/coolworld/fortyguard.py) and collector [`fortyguard_collect.py`](fortyguard_collect.py). The official Quickstart therefore is not claimed as project boilerplate.

AI tools were used during development for code review, debugging, test generation, documentation editing and figure-layout review. Final training, model evaluation, provider responses, thresholds and research figures are generated from the repository/Kaggle pipelines and checked against machine-readable artifacts. The deployed application does not use an AI assistant to fabricate temperatures, intervention effects or benchmark results.

## References

Amini, S., Huerta, A., Franke, J. *et al.* (2026) ‘Comprehensive compilation and quality assessment of street-level urban air temperature measurements across European networks’, *Scientific Data*, 13, 658. https://doi.org/10.1038/s41597-026-06804-4

Baek, D., Lee, G., Baek, J., Lee, H. and Ahn, S. (2026) ‘Learning to Theorize the World from Observation’, *Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)*, Oral.

Baek, J., Wu, Y.-F., Singh, G. and Ahn, S. (2025) ‘Dreamweaver: Learning Compositional World Models from Pixels’, *International Conference on Learning Representations (ICLR 2025)*. https://proceedings.iclr.cc/paper_files/paper/2025/hash/ae82a60c5ce50b5c4d18cfe3214eb684-Abstract-Conference.html

Kochkov, D., Yuval, J., Langmore, I. *et al.* (2024) ‘Neural general circulation models for weather and climate’, *Nature*, 632, 1060–1066. https://doi.org/10.1038/s41586-024-07744-y

Lam, R., Sanchez-Gonzalez, A., Willson, M. *et al.* (2023) ‘Learning skillful medium-range global weather forecasting’, *Science*, 382(6677), 1416–1421. https://doi.org/10.1126/science.adi2336

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L. and Long, M. (2024) ‘iTransformer: Inverted Transformers Are Effective for Time Series Forecasting’, *International Conference on Learning Representations (ICLR 2024)*.

Price, I., Sanchez-Gonzalez, A., Alet, F. *et al.* (2025) ‘Probabilistic weather forecasting with machine learning’, *Nature*, 637, 84–90. https://doi.org/10.1038/s41586-024-08252-9

Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J.Y. and Zhou, J. (2024) ‘TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting’, *International Conference on Learning Representations (ICLR 2024)*.

Balestriero, R. and LeCun, Y. (2025) ‘LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics’, *arXiv preprint* arXiv:2511.08544. **SIGReg originates here.**

Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y. and Balestriero, R. (2026) ‘LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels’, *arXiv preprint* arXiv:2603.19312.

## License

See [`LICENSE`](LICENSE).
