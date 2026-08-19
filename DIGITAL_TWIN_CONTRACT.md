# 3D Digital Twin Contract

The 3D interface is an evidence viewer and counterfactual decision surface. It must never blur observed measurements and modelled futures.

## A — OBSERVED
Displays only retrieved real evidence: FortyGuard thermal tiles and their real `average_temperature`; official cooling-intervention geometry; open map/building geometry; provenance/activity IDs/hashes.

Thermal extrusion may visually encode temperature. Its height is explicitly **not** atmospheric height.

## B — VALIDATED REPLAY
Replays a cooling intervention that actually happened. It requires an attributable site/date, real before/after observations, real control/matched observations or another declared causal design, and a stored evaluation manifest with uncertainty. Otherwise the UI returns `REPLAY_NOT_AVAILABLE`.

## C — COUNTERFACTUAL
Shows a proposed future intervention and is always labelled **MODELLED COUNTERFACTUAL**. It renders only if a validated checkpoint exists, the checkpoint hash matches its manifest, the manifest links real evidence, SAM returns an admissible identification set, and the action is within supported bounds.

Weak support -> `INSUFFICIENT_EVIDENCE` or a wider set. Mechanism/data contradiction -> `CONFLICT`. Missing model -> `MODEL_NOT_READY`.

## Scientific 3D rule
Buildings and intervention assets come from real geographic geometry. We do not invent a CFD volume or imply that colored columns are physical plumes. If a validated CFD/urban-energy solver is later integrated, its outputs get a separate model/provenance label.

## Outcome separation
Ambient air temperature, pavement/surface temperature, apparent/wet-bulb metrics, and radiant/thermal-comfort measures are not interchangeable. A cooler surface does not automatically imply lower human heat exposure.
