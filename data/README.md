# Data directory

No raw FortyGuard response, API key, private Kaggle bundle, trained checkpoint, or other empirical artifact is committed here.

Runtime layout:

- `../evidence/` — immutable locally collected API responses + provenance (git-ignored).
- `raw/` — optional licensed/downloaded real source files (git-ignored).
- `processed/urban_thermal_sequences.npz` — training bundle (git-ignored).
- `processed/urban_thermal_sequences.manifest.json` — provenance/schema manifest.

Before uploading any FortyGuard-derived data to Kaggle, cloud storage, or a public release, confirm that the applicable data-use terms permit that storage/redistribution. A free hackathon API key does not itself grant redistribution rights.

Action columns are physical coverage fractions in [0,1]: the fraction of each tile footprint affected by shade, added canopy, or reflective-pavement treatment. Non-zero actions must come from real mapped intervention records; they are never generated just to make action-conditioned training possible.
