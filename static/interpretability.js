(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  const STEPS = [
    {
      key: 'observe',
      title: '1 · OBSERVE REAL CITY HEAT',
      body: 'Load the immutable FortyGuard timeline. Every coloured polygon is one real provider tile inside the measured 36-tile AOI. Areas outside the polygon mask are not measured by this evidence bundle.',
      button: 'guideObserve',
      waitMs: 900,
    },
    {
      key: 'forecast',
      title: '2 · FORECAST THE NEXT 1–6 HOURS',
      body: 'Run the exact frozen seed-42 SAM-WM on the last 48 real hourly frames. The map becomes a model future, not a new observation. Use the bottom time slider to inspect +1 h through +6 h.',
      button: 'guideForecast',
      waitMs: 1400,
    },
    {
      key: 'prioritize',
      title: '3 · PRIORITIZE PERSISTENT FUTURE HOTSPOTS',
      body: 'Rank forecast tiles by mean future temperature and persistence in the selected hottest fraction across all six horizons. Yellow → orange → red is a relative priority ranking; hover/cards retain the true °C.',
      button: 'guideHotspots',
      waitMs: 1900,
    },
    {
      key: 'evidence',
      title: '4 · CHECK THE EVIDENCE BOUNDARY',
      body: 'Inspect Freiburg final-ID, Novi Sad zero-shot OOD-1, Turku zero-shot OOD-2, the promoted checkpoint, and the separate FortyGuard operational replay. A research forecast can be valid to inspect even when the stricter operational certification gate is not met.',
      button: 'guideEvidence',
      waitMs: 800,
    },
    {
      key: 'act',
      title: '5 · WHAT HAPPENS IN THE REAL WORLD?',
      body: 'A city engineer inspects a prioritized site, selects a feasible physical intervention, instruments treated and matched-control areas, measures pre/post temperature, and only then estimates a causal cooling effect. CoolWorld never invents that effect from a forecast alone.',
      button: null,
      waitMs: 0,
    },
  ];

  let currentStep = -1;

  function injectStylesheet() {
    if (document.querySelector('link[data-cw-interpretability]')) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/static/interpretability.css?v=1.0.0';
    link.dataset.cwInterpretability = 'true';
    document.head.appendChild(link);
  }

  function makeGuide() {
    const column = document.querySelector('.control-column');
    if (!column || $('cwGuide')) return;

    const guide = document.createElement('section');
    guide.id = 'cwGuide';
    guide.className = 'card cw-guide';
    guide.innerHTML = `
      <div class="cw-guide-head">
        <div>
          <div class="product-kicker">START HERE · GUIDED DEMO</div>
          <div class="card-title">WHAT COOLWORLD IS DOING — AND WHY</div>
        </div>
        <button id="cwRestart" class="tiny-action secondary" type="button">Restart</button>
      </div>
      <div class="cw-guide-intro">
        CoolWorld is a decision-support loop, not a magic cooling button: observe real heat → predict short-horizon evolution → prioritize where heat persists → physically intervene → measure the real effect.
      </div>
      <div class="cw-guide-actions">
        <button id="cwStart" class="primary" type="button">Start guided demo</button>
        <button id="cwPrev" class="secondary" type="button" disabled>← Previous</button>
        <button id="cwNext" class="primary" type="button" disabled>Next →</button>
      </div>
      <div id="cwGuideStep" class="cw-guide-step idle">
        <div class="cw-step-count">READY</div>
        <strong>Start with the real FortyGuard observations.</strong>
        <span>No live provider request is needed for the guided demo.</span>
      </div>
      <details class="cw-explain">
        <summary>Why SAM-WM is more than a static predictor or an LLM</summary>
        <div class="cw-explain-grid">
          <div><strong>48 h state context</strong><span>It conditions on a temporal city history rather than one isolated input.</span></div>
          <div><strong>Sparse physical graph</strong><span>Tiles interact through explicit local graph structure instead of an all-pairs black box.</span></div>
          <div><strong>Mechanism composition</strong><span>Conservative exchange, bounded source/sink, bounded residual, and optional wind transport are routed state-dependently.</span></div>
          <div><strong>World rollout</strong><span>The learned latent state is rolled forward recursively to +1…+6 h.</span></div>
          <div><strong>Uncertainty</strong><span>A frozen conformal radius is shown instead of presenting every forecast as certain.</span></div>
          <div><strong>Transfer + abstention</strong><span>Cross-city OOD tests are reported, and unsupported operational/causal claims remain locked.</span></div>
        </div>
        <p class="cw-boundary">This is a mechanism-structured world-model claim. The current artifacts do not establish human-child-level general intelligence or AGI.</p>
      </details>
    `;

    column.prepend(guide);
  }

  function addColourExplanation() {
    const legend = document.querySelector('.legend-card');
    if (!legend || $('cwColorMeaning')) return;
    const note = document.createElement('div');
    note.id = 'cwColorMeaning';
    note.className = 'cw-color-meaning';
    note.innerHTML = `
      <strong>How to read this coloured mask</strong>
      <span id="cwColorMeaningText">Each polygon is one real provider tile. The mask only covers the AOI for which recorded evidence exists.</span>
      <span class="cw-color-warning">Red on the thermal map means “warm end of the currently loaded field range”, not an automatic danger threshold.</span>
    `;
    legend.appendChild(note);
  }

  function addRealWorldLoop() {
    const hotspot = $('hotspotCardAnchor');
    if (!hotspot || $('cwRealWorldLoop')) return;
    const block = document.createElement('div');
    block.id = 'cwRealWorldLoop';
    block.className = 'cw-real-world-loop';
    block.innerHTML = `
      <strong>What to do after a hotspot is identified</strong>
      <ol>
        <li><b>Inspect:</b> verify the site, vulnerable users, ownership, geometry, shade, materials, and operational constraints.</li>
        <li><b>Choose:</b> select a feasible intervention candidate such as canopy, shade, reflective material, or another engineered action.</li>
        <li><b>Instrument:</b> define treated and matched-control areas before implementation.</li>
        <li><b>Measure:</b> collect pre/post temperature under comparable conditions.</li>
        <li><b>Validate:</b> estimate the causal effect; only evidence-backed effects may become action recommendations.</li>
      </ol>
    `;
    hotspot.appendChild(block);
  }

  function addGlossary() {
    const guide = $('cwGuide');
    if (!guide || $('cwGlossary')) return;
    const details = document.createElement('details');
    details.id = 'cwGlossary';
    details.className = 'cw-explain cw-glossary';
    details.innerHTML = `
      <summary>What the important words mean</summary>
      <dl>
        <dt>Observed</dt><dd>A recorded FortyGuard measurement returned by the provider API.</dd>
        <dt>SAM-WM forecast</dt><dd>A +1…+6 h model prediction generated from 48 h of real context. It is not observed truth.</dd>
        <dt>Hotspot priority</dt><dd>A relative ranking of forecast tiles. It tells an engineer where to investigate first, not how many degrees an intervention will cool.</dd>
        <dt>Operational certification</dt><dd>A stricter replay criterion for deployment. The frozen replay measured 79.8997% coverage against the pre-set 80.0000% minimum, so this version is not certified.</dd>
        <dt>Causal cooling effect</dt><dd>A measured intervention effect requiring treated/control evidence. It cannot be inferred from forecasting alone.</dd>
      </dl>
    `;
    guide.appendChild(details);
  }

  function updateColourMeaning() {
    const text = $('cwColorMeaningText');
    if (!text) return;
    const world = ($('worldStatus')?.textContent || '').toUpperCase();
    if (world.includes('RESEARCH FUTURE') || world.includes('PREDICT')) {
      text.textContent = 'Each polygon is one of the same 36 real provider-grid tiles, now coloured by the frozen SAM-WM forecast for the selected future hour. Use the time slider for +1…+6 h.';
    } else {
      text.textContent = 'Each polygon is one real FortyGuard provider tile. The mask only covers the 36-tile AOI returned by the recorded evidence; uncoloured city areas are outside this evidence bundle.';
    }
  }

  function normalizeReplayLanguage() {
    const strong = document.querySelector('#evidenceSummary .replay-gate strong');
    if (strong && /FAIL/i.test(strong.textContent || '')) {
      strong.textContent = 'Operational validation: NOT CERTIFIED · near threshold';
    }
    const gate = document.querySelector('#evidenceSummary .replay-gate');
    if (gate && !gate.querySelector('.cw-replay-explainer')) {
      const note = document.createElement('span');
      note.className = 'cw-replay-explainer';
      note.textContent = 'This is not an API crash and not a failed research forecast. It means empirical coverage was 79.8997% versus a pre-set 80.0000% deployment minimum; the threshold was not changed after evaluation.';
      gate.appendChild(note);
    }
  }

  function explainHotspotCards() {
    document.querySelectorAll('.hotspot-item').forEach((card) => {
      if (card.querySelector('.cw-why-hotspot')) return;
      const why = document.createElement('div');
      why.className = 'cw-why-hotspot';
      why.innerHTML = '<strong>Why this tile?</strong> Priority is based on the tile\'s mean SAM-WM temperature over +1…+6 h. “Persistent top-zone” is the share of forecast horizons in which it remains inside the selected hottest fraction.';
      card.appendChild(why);
    });
  }

  async function refreshOperationalStatus() {
    try {
      const response = await fetch('/api/product-status');
      if (!response.ok) return;
      const data = await response.json();
      const pill = $('statusOperational');
      const replay = data.provider_replay || {};
      if (pill && data.operational_certified === false) {
        const coverage = Number(replay.conformal_coverage);
        const minimum = Number(replay.minimum_required_coverage);
        if (Number.isFinite(coverage) && Number.isFinite(minimum)) {
          const gapPp = 100 * (minimum - coverage);
          pill.textContent = `operational validation pending · ${(100 * coverage).toFixed(2)}% / ${(100 * minimum).toFixed(2)}% · ${gapPp.toFixed(2)} pp short`;
          pill.title = 'Research forecast remains inspectable. This separate pre-set deployment gate was not met.';
        }
      }
    } catch {
      // Existing product layer owns endpoint error reporting.
    }
  }

  function updateGuideUi() {
    const panel = $('cwGuideStep');
    const prev = $('cwPrev');
    const next = $('cwNext');
    if (!panel || !prev || !next) return;

    if (currentStep < 0) {
      panel.className = 'cw-guide-step idle';
      panel.innerHTML = '<div class="cw-step-count">READY</div><strong>Start with the real FortyGuard observations.</strong><span>No live provider request is needed for the guided demo.</span>';
      prev.disabled = true;
      next.disabled = true;
      return;
    }

    const step = STEPS[currentStep];
    panel.className = `cw-guide-step active step-${step.key}`;
    panel.innerHTML = `<div class="cw-step-count">STEP ${currentStep + 1} OF ${STEPS.length}</div><strong>${step.title}</strong><span>${step.body}</span>`;
    prev.disabled = currentStep === 0;
    next.disabled = false;
    next.textContent = currentStep === STEPS.length - 1 ? 'Finish ✓' : 'Next →';
  }

  async function runStep(index) {
    if (index < 0 || index >= STEPS.length) return;
    currentStep = index;
    updateGuideUi();

    const step = STEPS[index];
    if (step.button) {
      $(step.button)?.click();
      if (step.waitMs) await sleep(step.waitMs);
    }

    if (step.key === 'observe') {
      document.querySelector('.map-stage')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (step.key === 'forecast') {
      $('forecastAnalyticsAnchor')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (step.key === 'prioritize') {
      $('hotspotCardAnchor')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (step.key === 'evidence') {
      $('evidenceCardAnchor')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (step.key === 'act') {
      $('cwRealWorldLoop')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  async function startGuide() {
    await runStep(0);
    $('cwNext').disabled = false;
  }

  function restartGuide() {
    currentStep = -1;
    updateGuideUi();
    $('guideObserve')?.click();
    window.setTimeout(() => {
      document.querySelector('.topbar')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 250);
  }

  function wireGuide() {
    $('cwStart')?.addEventListener('click', startGuide);
    $('cwRestart')?.addEventListener('click', restartGuide);
    $('cwPrev')?.addEventListener('click', () => {
      if (currentStep > 0) runStep(currentStep - 1);
    });
    $('cwNext')?.addEventListener('click', () => {
      if (currentStep < 0) return;
      if (currentStep >= STEPS.length - 1) {
        const panel = $('cwGuideStep');
        if (panel) {
          panel.className = 'cw-guide-step complete';
          panel.innerHTML = '<div class="cw-step-count">DEMO COMPLETE</div><strong>CoolWorld stops at an evidence-bounded engineering decision.</strong><span>Use Restart to replay the walkthrough, or inspect Advanced sections for developer/research details.</span>';
        }
        $('cwNext').disabled = true;
        return;
      }
      runStep(currentStep + 1);
    });
  }

  function simplifyPrimaryModes() {
    const replay = $('replayTab');
    if (replay) {
      replay.hidden = true;
      replay.setAttribute('aria-hidden', 'true');
      replay.tabIndex = -1;
    }
    const observed = document.querySelector('[data-mode="observed"]');
    const forecast = $('predictedTab');
    if (observed) observed.textContent = '1 · OBSERVE';
    if (forecast) forecast.textContent = '2 · FORECAST';
  }

  function observeDynamicPanels() {
    const evidence = $('evidenceSummary');
    if (evidence) {
      new MutationObserver(() => normalizeReplayLanguage())
        .observe(evidence, { childList: true, subtree: true });
    }
    const hotspots = $('hotspotList');
    if (hotspots) {
      new MutationObserver(() => explainHotspotCards())
        .observe(hotspots, { childList: true, subtree: true });
    }
    const world = $('worldStatus');
    if (world) {
      new MutationObserver(updateColourMeaning)
        .observe(world, { childList: true, characterData: true, subtree: true });
    }
  }

  async function init() {
    injectStylesheet();
    makeGuide();
    addColourExplanation();
    addRealWorldLoop();
    addGlossary();
    simplifyPrimaryModes();
    wireGuide();
    observeDynamicPanels();
    updateGuideUi();
    updateColourMeaning();
    await sleep(1200);
    normalizeReplayLanguage();
    explainHotspotCards();
    refreshOperationalStatus();

    // Keep first-time use deterministic: observed evidence is the starting state.
    if (($('frameLabel')?.textContent || '').trim().startsWith('0 frame')) {
      $('loadTimeline')?.click();
    }
  }

  window.addEventListener('load', () => {
    window.setTimeout(init, 1200);
  });
})();
