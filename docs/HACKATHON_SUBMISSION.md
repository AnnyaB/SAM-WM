# FortyGuard Hackathon'26 final submission checklist

**Hard deadline:** 30 August 2026, 11:59 PM GST. No late submissions.

## Four required deliverables

1. **Official submission form** — for a solo entry, submit the participant's own FortyGuard API key in the form only. The form also asks for the project title, one-line pitch, primary track, up to two optional secondary tags, target user/problem, city/area and time period, how the Temperature API was used, AI-tool disclosure, and the repository/live-demo/video links.
2. **Live demo** — a URL anyone can open in a private/incognito browser without login or installation. It must remain live through judging.
3. **Demo video** — maximum 3 minutes; YouTube or Loom; unlisted is acceptable. It must show the **working project**, not slides. Voiceover is required; face is optional.
4. **Code repository** — GitHub or GitLab. If private, add `Hackathon-FG` / `hackathon@fortyguard.com`. The README must show how to run from scratch, what does not work yet, and one real FortyGuard API request + response. **Never commit an API key.**

Official form: https://forms.gle/jLgBzVTG1NhJ3gNe6

## Track-specific requirement that must not be skipped

FortyGuard's final `#help-general` guidance on 30 August adds one requirement beyond the four generic deliverables: **each track has its own demo requirement plus one proof item**, defined in the Participant Handbook. The shared Slack canvas gives the Track 1 examples and judging rubric but does not reproduce that proof-item text. Therefore the Track 1 handbook section must be checked manually before final submission; do not assume the generic checklist alone completes Track 1 compliance.

For SAM-WM, the primary track remains **Track 1 — Resilient Cities & Infrastructure**. The product evidence already foregrounds a Track 1 use case — persistent-hotspot ranking for engineering review — but the final video/form must also satisfy the handbook's exact Track 1 demo/proof wording once confirmed.

## Judging weights

| Criterion | Weight | SAM-WM evidence to foreground |
|---|---:|---|
| Impact & Relevance | 40% | real San José thermal evidence; persistent-hotspot prioritization for resilient-city engineering review; explicit path from forecast → field action → measured treated/control evidence |
| Technical Execution | 35% | real FastAPI/3D deployment; custom fail-closed FortyGuard integration; immutable request/response hashes; 40-run matched suite; zero-shot OOD; uncertainty; tests and frozen deployment gates |
| Innovation | 15% | sparse mechanism-structured thermal world model with typed exchange/transport/source/residual operators, adaptive routing and uncertainty-aware rollout |
| Communication | 10% | concise README; real API proof; honest failure/claim boundaries; ≤3-minute working-product video with voiceover |

## Submission framing

- **Title:** SAM-WM · CoolWorld
- **One-line pitch:** A sparse mechanism-structured world model that turns real FortyGuard temperature evidence into uncertainty-aware six-hour urban heat forecasts and persistent-hotspot priorities for resilient-city engineering review.
- **Primary track:** Track 1 — Resilient Cities & Infrastructure.
- **Optional secondary tag:** Track 5 — Model Designing.
- **Deployment use case:** San José, California; 65 recorded hourly FortyGuard TCM frames on one 36-tile grid.
- **Research validation:** Freiburg held-out + zero-shot Novi Sad. These are research validation domains, not the deployed U.S. provider area.

## Boilerplate / provenance clarification

FortyGuard allows the official Temperature API Quickstart as pre-existing boilerplate if it is disclosed in the README. **SAM-WM was not cloned from that Quickstart.** The repository was created on 18 August 2026 and is not a fork; the Temperature API integration is implemented by the custom client in `src/coolworld/fortyguard.py` and the bounded collector in `fortyguard_collect.py`. Do not falsely add a Quickstart-boilerplate attribution. If any local code was in fact copied from the Quickstart outside Git history, disclose that before submission rather than relying on this repository-level audit.

## Final pre-submit checks

- [ ] Open the Participant Handbook and confirm the exact **Track 1 demo requirement + proof item**; make both visible in the final demo/video/form.
- [ ] Open `https://sam-wm-coolworld.onrender.com` in a fresh private/incognito browser: no login, no installation.
- [ ] Confirm `/api/health` returns `status: ok` and the Observe → Forecast → Prioritize → Evidence workflow renders.
- [ ] Confirm the app clearly labels the SAM-WM output as a **research forecast** when the fixed operational replay gate has not passed.
- [ ] Record the actual product, not slides; keep the video ≤3:00 and use voiceover.
- [ ] Keep the FortyGuard API key out of Git, video, screenshots, URLs, browser code and chat; enter it only in the official form.
- [ ] If the repository remains private, add `Hackathon-FG` (`hackathon@fortyguard.com`) before submission.
- [ ] Confirm README contains run-from-scratch commands, limitations, an inline real API request/response example, AI-tool disclosure, exact results and the live-demo link.
- [ ] Submit repository, live demo and video links before the hard deadline.
- [ ] Re-test the Render URL after submission; free services can sleep.
