# Global-scale architecture

CoolWorld-SAM is geographically portable, while the **hackathon deployment remains U.S.-only**.

The core is provider-driven rather than Phoenix-specific:
- `TemperatureProvider` — observed/forecast hyperlocal thermal field.
- `UrbanGeometryProvider` — real buildings/roads/assets.
- `InterventionRegistry` — real cooling projects with type/timing.
- `ExposureProvider` — population/transit/pedestrian/vulnerability evidence.
- `CounterfactualModel` — calibrated intervention world model.
- `Planner` — robust intervention selection under constraints.

Phoenix is one implementation. A new region is enabled only when equivalent evidence providers exist and the model is revalidated/calibrated there. Global deployment is earned by transfer evidence, not by renaming `city` to `location`.
