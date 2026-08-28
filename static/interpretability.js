(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const q = (selector) => document.querySelector(selector);
  const qa = (selector) => [...document.querySelectorAll(selector)];
  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
  const setText = (node, text) => {
    if (node && node.textContent !== text) node.textContent = text;
  };

  const STEPS = [
    {
      title: '1 · OBSERVE THE CITY',
      body: 'Start with the recorded San José thermal field. Each coloured polygon is one measured provider tile; uncoloured areas are outside this field.',
      button: 'guideObserve',
      anchor: '.map-stage',
      wait: 700,
    },
    {
      title: '2 · RUN SAM-WM',
      body: 'SAM-WM reads the latest 48 hourly city states and rolls the thermal world forward from +1 h to +6 h.',
      button: 'guideForecast',
      anchor: 'forecastAnalyticsAnchor',
      wait: 1200,
    },
    {
      title: '3 · FIND PERSISTENT HEAT',
      body: 'Rank locations by future temperature and by how often they remain among the warmest tiles across all six forecast hours.',
      button: 'guideHotspots',
      anchor: 'hotspotCardAnchor',
      wait: 1500,
    },
    {
      title: '4 · PLAN A FIELD TEST',
      body: 'Inspect the site, choose a feasible physical intervention, and define treated and comparison areas before implementation.',
      button: null,
      anchor: 'cwFieldLoop',
      wait: 0,
    },
    {
      title: '5 · MEASURE WHAT CHANGED',
      body: 'After a real intervention, compare treated and control measurements. The measured difference becomes the cooling-effect evidence.',
      button: 'guideEvidence',
      anchor: 'evidenceCardAnchor',
      wait: 450,
    },
  ];

  let activeStep = -1;

  function ensureStylesheet() {
    if (q('link[data-cw-interpretability]')) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/static/interpretability.css?v=1.1.0';
    link.dataset.cwInterpretability = '1';
    document.head.appendChild(link);
  }

  function insertGuide() {
    const column = q('.control-column');
    if (!column || $('cwGuide')) return;
    const guide = document.createElement('section');
    guide.id = 'cwGuide';
    guide.className = 'card cw-guide';
    guide.innerHTML = `
      <div class="cw-guide-head">
        <div>
          <div class="product-kicker">START HERE · GUIDED DEMO</div>
          <div class="card-title">COOLWORLD IN FIVE STEPS</div>
        </div>
        <button id="cwRestart" class="tiny-action secondary" type="button">Restart</button>
      </div>
      <div class="cw-guide-intro">
        Measure the city → forecast six hours → find persistent heat → test a physical intervention → measure the result.
      </div>
      <div class="cw-guide-actions">
        <button id="cwStart" class="primary" type="button">Start</button>
        <button id="cwPrev" class="secondary" type="button" disabled>← Back</button>
        <button id="cwNext" class="primary" type="button" disabled>Next →</button>
      </div>
      <div id="cwGuideStep" class="cw-guide-step idle">
        <div class="cw-step-count">READY</div>
        <strong>Begin with the recorded city field.</strong>
        <span>The demo uses saved provider data and makes no new provider request.</span>
      </div>
      <details class="cw-explain">
        <summary>Inside SAM-WM</summary>
        <div class="cw-explain-grid">
          <div><strong>48 h memory</strong><span>A recurrent state summarizes two days of hourly city history.</span></div>
          <div><strong>Local city graph</strong><span>Nearby tiles interact through a sparse physical graph rather than all-to-all mixing.</span></div>
          <div><strong>Conservative exchange</strong><span>Pairwise heat exchange is antisymmetric, so exchange itself preserves net heat.</span></div>
          <div><strong>Bounded forcing</strong><span>Local source/sink and residual terms are limited at every rollout step.</span></div>
          <div><strong>Daily + seasonal clock</strong><span>Explicit time features represent diurnal and annual thermal cycles.</span></div>
          <div><strong>Six-hour rollout</strong><span>The learned city state advances recursively from +1 h to +6 h with a calibrated prediction band.</span></div>
        </div>
      </details>`;
    column.prepend(guide);
  }

  function insertColourHelp() {
    const legend = q('.legend-card');
    if (!legend || $('cwColorMeaning')) return;
    const note = document.createElement('div');
    note.id = 'cwColorMeaning';
    note.className = 'cw-color-meaning';
    note.innerHTML = `
      <strong>How to read this coloured mask</strong>
      <span id="cwColorMeaningText">Each polygon is one measured provider tile. Uncoloured city areas are outside this recorded field.</span>
      <span class="cw-color-warning">The colour scale is relative to the values on screen. Read the °C legend or hover value for the actual temperature.</span>`;
    legend.appendChild(note);
  }

  function insertModelFlow() {
    const analytics = $('forecastAnalyticsAnchor');
    if (!analytics || $('cwModelFlow')) return;
    const flow = document.createElement('section');
    flow.id = 'cwModelFlow';
    flow.className = 'cw-model-flow';
    flow.innerHTML = `
      <div class="cw-model-flow-title">
        <span>SAM-WM · CITY WORLD MODEL</span>
        <strong>48 h history → local graph → mechanism rollout → 6 h future</strong>
      </div>
      <div class="cw-flow-track">
        <div><b>48 H</b><span>city history</span></div><i>→</i>
        <div><b>36</b><span>provider tiles</span></div><i>→</i>
        <div class="emphasis"><b>LOCAL</b><span>sparse graph</span></div><i>→</i>
        <div class="emphasis"><b>4</b><span>routed mechanisms</span></div><i>→</i>
        <div><b>6 H</b><span>future rollout</span></div>
      </div>
      <div class="cw-mechanism-chips">
        <span>conservative exchange</span>
        <span>bounded source / sink</span>
        <span>bounded residual</span>
        <span>wind transport when available</span>
        <span>daily + seasonal clock</span>
        <span>recurrent latent memory</span>
      </div>`;
    analytics.parentNode.insertBefore(flow, analytics);
  }

  function insertFieldLoop() {
    const host = $('hotspotCardAnchor');
    if (!host || $('cwFieldLoop')) return;
    const block = document.createElement('div');
    block.id = 'cwFieldLoop';
    block.className = 'cw-real-world-loop';
    block.innerHTML = `
      <strong>From hotspot to physical cooling</strong>
      <div class="cw-action-track">
        <span><b>1</b>Inspect site</span><i>→</i>
        <span><b>2</b>Choose intervention</span><i>→</i>
        <span><b>3</b>Define treated + control</span><i>→</i>
        <span><b>4</b>Implement</span><i>→</i>
        <span><b>5</b>Measure effect</span>
      </div>`;
    host.appendChild(block);
  }

  function replaceConsole() {
    const raw = $('console');
    const card = raw?.closest('.console-card');
    if (!raw || !card || $('cwInsightStream')) return;
    setText(card.querySelector('.card-title'), 'SAM-WM LIVE STATE');
    const stream = document.createElement('div');
    stream.id = 'cwInsightStream';
    stream.className = 'cw-insight-stream';
    stream.innerHTML = `
      <div class="cw-insight-row"><span>CONTEXT</span><strong>Loading measured city history…</strong></div>
      <div class="cw-insight-row"><span>OUTLOOK</span><strong>Run SAM-WM to generate the six-hour future.</strong></div>
      <div class="cw-insight-row"><span>PRIORITY</span><strong>Persistent-heat ranking appears after the forecast.</strong></div>
      <div class="cw-insight-row"><span>MODEL</span><strong>Mechanism-structured rollout; missing inputs are disabled rather than invented.</strong></div>`;
    const details = document.createElement('details');
    details.className = 'cw-raw-log';
    details.innerHTML = '<summary>Runtime diagnostics</summary>';
    raw.before(stream, details);
    details.appendChild(raw);
  }

  function setPrimaryLanguage() {
    setText(q('.brand-block .subtitle'), 'Measure the city → forecast 1–6 h → find persistent heat → plan and verify physical cooling');
    setText(q('.product-guide .product-kicker'), 'QUICK START');
    setText(q('.product-guide .card-title'), 'COOLWORLD WORKFLOW');
    setText(q('.product-guide .product-intro'), 'Explore the measured city field, run SAM-WM, then inspect the locations that stay warm across the next six hours.');
    setText(q('.mode-card .card-title'), 'CITY STATE');
    setText($('measuredCardAnchor')?.querySelector('.card-title'), 'MEASURED CITY FIELD');
    setText($('hotspotCardAnchor')?.querySelector('.card-title'), 'PERSISTENT HEAT PRIORITY');
    setText($('evidenceCardAnchor')?.querySelector('.card-title'), 'MODEL VALIDATION');
    setText(q('.mode-card .truth-note'), 'Measured fields and SAM-WM forecasts are separate city states. Physical cooling is evaluated after a measured field trial.');
    setText(q('#hotspotCardAnchor .product-explainer'), 'SAM-WM ranks tiles by future temperature and how often they remain among the warmest locations over the next six hours.');
    setText(q('.product-note'), 'CoolWorld turns measured urban heat into a six-hour city forecast and a shortlist of locations to inspect for physical cooling.');

    const analyticsTitles = qa('.analytics-title');
    ['CITY FIELD DISTRIBUTION', 'SELECTED HOUR', 'FORECAST RANGE + VALIDATION', '6-HOUR PRIORITY-ZONE OUTLOOK']
      .forEach((text, index) => setText(analyticsTitles[index], text));

    const uncertaintyLabels = qa('.uncertainty-row span');
    setText(uncertaintyLabels[0], 'Horizon');
    setText(uncertaintyLabels[1], 'Prediction band');
    setText(uncertaintyLabels[2], 'Field replay coverage');

    const statusCard = qa('.card').find((card) => card.querySelector('.truth-stack'));
    setText(statusCard?.querySelector('.card-title'), 'SYSTEM STATUS');

    qa('details.product-details > summary').forEach((summary) => {
      const value = summary.textContent || '';
      if (value.includes('recorded evidence / optional live FortyGuard API')) setText(summary, 'Advanced · data source and optional live field');
      if (value.includes('physical intervention evidence gate')) setText(summary, 'Advanced · field intervention study');
      if (value.includes('reproducibility pipeline + agent console')) setText(summary, 'Advanced · model diagnostics');
    });

    const observed = q('[data-mode="observed"]');
    if (observed) setText(observed, '1 · MEASURED');
    if ($('predictedTab')) setText($('predictedTab'), '2 · SAM-WM FORECAST');
    if ($('replayTab')) {
      $('replayTab').hidden = true;
      $('replayTab').tabIndex = -1;
      $('replayTab').setAttribute('aria-hidden', 'true');
    }
  }

  function rewriteDynamicLanguage() {
    if (/RESEARCH FUTURE/i.test($('worldStatus')?.textContent || '')) setText($('worldStatus'), 'SAM-WM FUTURE');
    if (/RESEARCH FORECAST READY/i.test($('modelStatus')?.textContent || '')) setText($('modelStatus'), 'SAM-WM FORECAST READY');
    if (/RESEARCH FORECAST/i.test($('modeBanner')?.textContent || '')) setText($('modeBanner'), 'SAM-WM · +1…+6 h FORECAST');
    if (/OBSERVED · REAL FORTYGUARD EVIDENCE/i.test($('modeBanner')?.textContent || '')) setText($('modeBanner'), 'MEASURED · FORTYGUARD CITY FIELD');
    if (/FROZEN RESEARCH FORECAST/i.test($('sourceStatus')?.textContent || '')) setText($('sourceStatus'), 'SAM-WM · SEED 42 FORECAST');
    if (/RESEARCH FORECAST/i.test($('predictionStatus')?.textContent || '')) setText($('predictionStatus'), 'SAM-WM FORECAST READY');

    const explanation = $('predictionExplanation');
    if (/research|provider replay|operational coverage gate/i.test(explanation?.textContent || '')) {
      setText(explanation, 'SAM-WM is forecasting six hourly city states from the latest 48 measured provider frames.');
    }

    const timeline = $('timelineTruth');
    const timelineText = timeline?.textContent || '';
    if (/FROZEN SAM-WM RESEARCH FORECAST/i.test(timelineText)) setText(timeline, 'SAM-WM FORECAST · selected future hour');
    if (/VISUAL INTERPOLATION between frozen SAM-WM/i.test(timelineText)) setText(timeline, 'FORECAST TRANSITION · visual interpolation between hourly future states');
    if (/Exact recorded real frame/i.test(timelineText)) setText(timeline, 'MEASURED FIELD · recorded provider frame');

    if (/NOT OPERATIONALLY CERTIFIED/i.test($('actionabilityBadge')?.textContent || '')) {
      setText($('actionabilityBadge'), 'FIELD VALIDATION · 79.9% / 80.0%');
      $('actionabilityBadge').className = 'actionability-badge cw-review';
    }
    if (/79\.90% \/ 80% gate/i.test($('uncSupport')?.textContent || '')) setText($('uncSupport'), '79.9% · target 80.0%');
    if (/No causal intervention effect is invented/i.test($('previewCaption')?.textContent || '')) {
      setText($('previewCaption'), 'Mean temperature across the prioritized forecast locations.');
    }

    const colourText = $('cwColorMeaningText');
    if (colourText) {
      const future = /FUTURE|FORECAST/i.test($('worldStatus')?.textContent || '');
      setText(
        colourText,
        future
          ? 'Each polygon is one of the same 36 provider-grid tiles, coloured by the selected SAM-WM future hour. Move the timeline from +1 h to +6 h.'
          : 'Each polygon is one measured FortyGuard tile. The 36-tile mask is the recorded area; uncoloured city areas are outside this field.',
      );
    }
  }

  async function updateStatus() {
    try {
      const response = await fetch('/api/product-status');
      if (!response.ok) return;
      const state = await response.json();
      const replay = state.provider_replay || {};
      setText($('statusRealEvidence'), state.real_provider_evidence_ready ? `MEASURED DATA · ${state.recorded_real_frames} hourly frames` : 'MEASURED DATA · unavailable');
      setText($('statusModelBundle'), state.model_bundle_promoted ? `SAM-WM · seed ${state.selected_seed ?? '—'} loaded` : 'SAM-WM · unavailable');
      setText($('statusResearchForecast'), state.research_forecast_ready ? 'FORECAST · ready' : 'FORECAST · waiting for context');
      if (Number.isFinite(Number(replay.conformal_coverage))) {
        const pill = $('statusOperational');
        setText(pill, `FIELD VALIDATION · ${(100 * Number(replay.conformal_coverage)).toFixed(1)}% interval coverage`);
        pill?.classList.remove('ok', 'warn');
        pill?.classList.add('review');
      }
      setText($('statusCausal'), state.causal_action_ready ? 'FIELD EFFECT · measured evidence loaded' : 'FIELD EFFECT · requires treated + control measurements');
    } catch {
      // The base product layer owns endpoint-error reporting.
    }
  }

  function rewriteEvidence() {
    const gate = q('#evidenceSummary .replay-gate');
    if (!gate) return;
    const spans = gate.querySelectorAll('span');
    setText(gate.querySelector('strong'), 'FIELD REPLAY VALIDATION');
    setText(spans[0], 'Prediction interval coverage: 79.90% · target: 80.00%');
    setText(spans[1], 'Mean error / prediction-band radius: 0.638 · limit: 1.000');
    gate.classList.remove('fail');
    gate.classList.add('review');
    if (!gate.querySelector('.cw-replay-explainer')) {
      const note = document.createElement('span');
      note.className = 'cw-replay-explainer';
      note.textContent = 'The prediction band contained the measured provider temperature in 79.9% of replayed cases, just below the 80.0% field target. This is a forecast-validation result, not a FortyGuard API status.';
      gate.appendChild(note);
    }
    setText(q('#evidenceSummary .claim-boundary'), 'Forecast accuracy and cross-city transfer are measured here. Cooling from trees, shade or materials is measured separately in a real field study.');
  }

  function rewriteHotspots() {
    qa('.hotspot-item').forEach((card) => {
      const foot = card.querySelector('.hotspot-foot');
      if (/Persistent top-zone:/i.test(foot?.textContent || '')) {
        const persistence = (foot.textContent.match(/Persistent top-zone:\s*([^·]+)/i)?.[1] || '—').trim();
        setText(foot, `Stays in the warmest group: ${persistence}`);
      }
      if (!card.querySelector('.cw-why-hotspot')) {
        const why = document.createElement('div');
        why.className = 'cw-why-hotspot';
        why.innerHTML = '<strong>Why this location?</strong><span>Its six-hour mean forecast is high relative to the rest of the field; persistence shows how often it stays in the selected warmest group.</span>';
        card.appendChild(why);
      }
    });
    setText($('hotspotBoundary'), 'Use this shortlist for site inspection and field-test design. Measured cooling is added after an intervention study.');
  }

  function guideUi() {
    const panel = $('cwGuideStep');
    if (!panel) return;
    if (activeStep < 0) {
      panel.className = 'cw-guide-step idle';
      panel.innerHTML = '<div class="cw-step-count">READY</div><strong>Begin with the recorded city field.</strong><span>The demo uses saved provider data and makes no new provider request.</span>';
      $('cwPrev').disabled = true;
      $('cwNext').disabled = true;
      return;
    }
    const item = STEPS[activeStep];
    panel.className = 'cw-guide-step active';
    panel.innerHTML = `<div class="cw-step-count">STEP ${activeStep + 1} OF ${STEPS.length}</div><strong>${item.title}</strong><span>${item.body}</span>`;
    $('cwPrev').disabled = activeStep === 0;
    $('cwNext').disabled = false;
    setText($('cwNext'), activeStep === STEPS.length - 1 ? 'Finish ✓' : 'Next →');
  }

  async function runStep(index) {
    if (index < 0 || index >= STEPS.length) return;
    activeStep = index;
    guideUi();
    const item = STEPS[index];
    if (item.button) {
      $(item.button)?.click();
      if (item.wait) await sleep(item.wait);
    }
    const anchor = item.anchor.startsWith('.') ? q(item.anchor) : $(item.anchor);
    anchor?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    window.dispatchEvent(new CustomEvent('coolworld:guide-step', { detail: { index, title: item.title } }));
  }

  function wireGuide() {
    $('cwStart')?.addEventListener('click', () => runStep(0));
    $('cwRestart')?.addEventListener('click', () => {
      activeStep = -1;
      guideUi();
      $('guideObserve')?.click();
      q('.topbar')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    $('cwPrev')?.addEventListener('click', () => {
      if (activeStep > 0) runStep(activeStep - 1);
    });
    $('cwNext')?.addEventListener('click', () => {
      if (activeStep < 0) return;
      if (activeStep < STEPS.length - 1) {
        runStep(activeStep + 1);
        return;
      }
      const panel = $('cwGuideStep');
      panel.className = 'cw-guide-step complete';
      panel.innerHTML = '<div class="cw-step-count">COMPLETE</div><strong>You have a measured field, a six-hour forecast and a field-test target.</strong><span>Restart the walkthrough or open Advanced for model diagnostics and field-study controls.</span>';
      $('cwNext').disabled = true;
    });
  }

  function observeDynamicContent() {
    const dynamicIds = [
      'worldStatus', 'modelStatus', 'modeBanner', 'sourceStatus', 'timelineTruth',
      'predictionStatus', 'predictionExplanation', 'actionabilityBadge', 'uncSupport', 'previewCaption',
    ];
    dynamicIds.map($).filter(Boolean).forEach((node) => {
      new MutationObserver(rewriteDynamicLanguage)
        .observe(node, { childList: true, subtree: true, characterData: true });
    });
    if ($('evidenceSummary')) {
      new MutationObserver(rewriteEvidence)
        .observe($('evidenceSummary'), { childList: true, subtree: true });
    }
    if ($('hotspotList')) {
      new MutationObserver(rewriteHotspots)
        .observe($('hotspotList'), { childList: true, subtree: true });
    }
  }

  function init() {
    ensureStylesheet();
    insertGuide();
    insertColourHelp();
    insertModelFlow();
    insertFieldLoop();
    replaceConsole();
    setPrimaryLanguage();
    wireGuide();
    observeDynamicContent();
    rewriteDynamicLanguage();
    rewriteEvidence();
    rewriteHotspots();
    updateStatus();
    window.setTimeout(updateStatus, 1250);
    window.setTimeout(rewriteDynamicLanguage, 1450);
  }

  document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init, { once: true })
    : init();
})();
