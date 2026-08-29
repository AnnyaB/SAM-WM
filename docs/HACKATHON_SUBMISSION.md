# FortyGuard Hackathon'26 submission checklist

Deadline: **30 August 2026, 11:59 PM GST**.

## Required submission pieces

1. **Official submission form** — project title, one-line pitch, primary track, optional secondary tags, target user/problem, location/time period, how the FortyGuard API was used, API key(s) entered in the form only, AI-tool disclosure, repository link, live-demo link, and demo-video link.
2. **Live demo** — must open in a private/incognito browser with no login or install and remain available through judging.
3. **Demo video** — maximum 3 minutes; YouTube or Loom is acceptable; show the actual working project, not slides; voiceover is required; face is optional.
4. **Code repository** — GitHub or GitLab. If private, add `Hackathon-FG` / `hackathon@fortyguard.com` as collaborator. README must show how to run from scratch, what does not work yet, and one real FortyGuard API request + response. Never commit API keys.

Official form: https://forms.gle/jLgBzVTG1NhJ3gNe6

## Judging weights

| Criterion | Weight | SAM-WM evidence to foreground |
|---|---:|---|
| Impact & Relevance | 40% | persistent urban hotspot prioritization, real provider data, field-intervention workflow, explicit client/user path |
| Technical Execution | 35% | real FastAPI/3D deployment, immutable provider evidence, 40-run paper suite, zero-shot OOD, calibration, tests/CI, fail-closed gates |
| Innovation | 15% | sparse mechanism-structured world model, bounded thermal operators, uncertainty/evidence separation |
| Communication | 10% | clean README, ≤3 min working-product video with voiceover, explicit claim boundaries |

## Recommended submission framing

- **Title:** SAM-WM · CoolWorld
- **One-line pitch:** A sparse mechanism-structured world model that turns real FortyGuard temperature evidence into uncertainty-aware six-hour urban heat forecasts and persistent-hotspot priorities for resilient-city engineering review.
- **Primary track:** Track 1 — Resilient Cities & Infrastructure
- **Secondary tag:** Track 5 — Model Designing
- **Deployment use case:** San José, California, using recorded FortyGuard TCM evidence on a 36-tile grid.
- **Research validation:** Freiburg held-out + zero-shot Novi Sad; these research cities are validation domains, not the deployed U.S. provider AOI.

## Final pre-submit checks

- [ ] Open `https://sam-wm-coolworld.onrender.com` in a private/incognito browser: no login, no install.
- [ ] Confirm `/api/health` is healthy and the guided Observe → Forecast → Prioritize → Evidence flow works.
- [ ] Record the actual working product, not slides; keep video ≤3:00 and include voiceover.
- [ ] Keep the FortyGuard API key out of Git, video, screenshots, browser code, and chat; enter it only in the official form where requested.
- [ ] If the repository remains private, add `Hackathon-FG` (`hackathon@fortyguard.com`) before submission.
- [ ] Confirm README includes run-from-scratch instructions, limitations, real API request/response, AI-tool disclosure, results, and live-demo link.
- [ ] Submit repository, live demo, and video links in the official form before the deadline.
- [ ] Re-test the free Render deployment after submission because free services may sleep.
