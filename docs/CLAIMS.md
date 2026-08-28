# Scientific and product claim boundary

This file is a compact truth contract for papers, demos, README text, UI labels, and presentations.

## Supported by current artifacts

The current frozen evidence supports saying that SAM-WM / CoolWorld:

- forecasts real urban thermal fields;
- performs +1…+6 h multi-step rollout;
- was evaluated on Freiburg final held-out data;
- was evaluated zero-shot on Novi Sad without target fine-tuning or OOD recalibration;
- was evaluated zero-shot on preregistered Turku / FAIRUrbTemp without target fine-tuning or OOD recalibration;
- uses real recorded FortyGuard TCM evidence for a San José provider-grid runtime demonstration;
- exposes calibrated uncertainty inherited from the frozen Freiburg validation calibration;
- handles unavailable modalities explicitly rather than fabricating observations;
- ranks forecast-persistent hotspots as non-causal decision support;
- preserves model/evidence provenance and fails closed when required evidence is absent;
- uses a compact 117,705-parameter frozen model in the reported runs.

## Supported with an important qualifier

It is valid to say:

> The same frozen SAM-WM was replayed on recorded FortyGuard data and achieved provider-domain MAE 2.0475 °C with MAE/radius ratio 0.6375, but the fixed operational coverage gate failed narrowly: 79.8997% coverage versus the preregistered 80% minimum.

It is **not** valid to shorten that into “operationally certified.”

## Not supported by current artifacts

Do not claim:

- human-child-level general intelligence;
- AGI or ASI;
- universal or benchmark-wide SOTA superiority;
- superiority over MLML, Dreamweaver, NEO, JEPA, Dreamer, LeWorldModel, or another model without an executed comparator protocol;
- a causal cooling effect from an observational forecast;
- that planting a tree, adding shade, or using reflective pavement will cool a particular tile by a specific number of degrees unless a valid treated/control action artifact supports it;
- that the software itself physically cools Earth;
- planetary-scale validation;
- guaranteed life-saving impact from the current experiment;
- guaranteed hackathon score, prize, paper acceptance, or lab admission.

## Correct intervention wording

Preferred:

> CoolWorld identifies forecast-persistent urban heat priorities and proposes physical intervention categories for engineering investigation. Cooling-effect magnitude remains unavailable until independent treated/control evidence is validated.

Avoid:

> CoolWorld predicts that tree canopy will cool this tile by 2.3 °C.

unless that exact value is supported by a valid action-evidence artifact under the request conditions.

## Correct intelligence wording

Preferred:

> Mechanism-structured predictive world modeling with cross-city zero-shot evaluation, uncertainty awareness, explicit missing-modality handling, and evidence-triggered abstention.

Avoid:

> Human-like or child-like general intelligence.

The former is testable from the current system; the latter is not established by these experiments.

## Correct Earth-cooling wording

Preferred:

> The system is an urban thermal decision-support layer in a physical cooling loop: observe → forecast → prioritize → physically intervene → measure → validate.

Avoid:

> The neural network cools Earth.

Physical interventions perform the cooling. The software helps decide where evidence says investigation is useful and records when evidence is insufficient.
