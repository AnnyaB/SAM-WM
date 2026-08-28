# ≤3-minute CoolWorld judge demo

The goal is to demonstrate one coherent product story, not every developer control.

## 0:00–0:20 — problem and promise

Say:

> Cities need to know not only where heat is now, but where it is likely to persist over the next few hours. CoolWorld combines real FortyGuard thermal evidence with our frozen SAM-WM world model to forecast the city and prioritize future hotspots, while refusing to invent unsupported cooling effects.

Show the main 3D city view and the four-step guide.

## 0:20–0:50 — Observe real evidence

Click **1 · OBSERVE**.

Show:

- 65 recorded real provider frames;
- the 36-tile San José grid;
- real 3D basemap/buildings;
- observed ground thermal field;
- timestamp / activity / content hash;
- histogram and mean/P95/max/min;
- timeline playback.

Say explicitly that between-frame animation is visual interpolation, not an invented measurement.

## 0:50–1:30 — Forecast with the frozen SAM-WM

Click **2 · FORECAST**.

Show:

- +1…+6 h model frames;
- the 3D thermal field changing through the forecast horizon;
- SAM-WM mean future trajectory;
- conformal uncertainty;
- model state `research forecast ready`.

Say:

> This is the exact seed-42 frozen checkpoint selected before final/OOD evaluation. The output is labelled model prediction, not observation.

Do not claim operational certification; the fixed provider replay coverage gate narrowly failed.

## 1:30–2:05 — Prioritize future hotspots

Click **3 · PRIORITIZE**.

Show:

- the yellow→orange→red relative-priority 3D view;
- top hotspot cards;
- current true °C;
- +6 h predicted °C;
- future maximum;
- persistence across horizons;
- uncertainty radius.

Say:

> The red/yellow view is relative forecast priority, not fake absolute temperature. It tells an engineer which locations remain hottest in the model future. We suggest physical action categories such as trees, shade, or reflective surfaces, but we do not invent a cooling delta.

## 2:05–2:35 — Evidence and abstention

Click **4 · EVIDENCE**.

Show:

- Freiburg final ID result;
- Novi Sad zero-shot OOD-1;
- Turku zero-shot OOD-2;
- 117,705-parameter frozen model;
- FortyGuard operational replay `FAIL` at 79.8997% vs fixed 80% coverage threshold;
- MAE/radius criterion passing.

Say:

> We preserve the failure rather than moving the threshold. The research forecast is available, but operational certification and causal intervention effects remain blocked.

## 2:35–2:55 — real-world cooling loop

Return briefly to the four-step workflow.

Say:

> CoolWorld is the intelligence layer in a physical loop: observe, forecast, prioritize, deploy a real intervention, measure treated versus control outcomes, and only then validate a cooling effect. The physical intervention cools the city; the software helps place evidence and uncertainty around that decision.

## 2:55–3:00 — finish

Say:

> One frozen model, real provider evidence, cross-city zero-shot evaluation, interpretable 3D forecasting, and explicit abstention when evidence is insufficient.

Stop the recording.

## Recording rules

- Keep the browser full screen and zoom so the 3D map and right-side workflow are readable.
- Use voiceover; face camera is unnecessary unless the submission specifically asks for it.
- Do not show API keys, terminal secrets, or private repository settings.
- Do not trigger a new live provider request during the demo.
- Do not spend time opening advanced diagnostics unless a judge asks.
- Keep the operational replay failure visible and explain it once; do not hide it.
