(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const q = (selector) => document.querySelector(selector);

  function ensureStylesheet() {
    if (q('link[data-cw-city-model]')) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/static/city-model.css?v=1.0.0';
    link.dataset.cwCityModel = '1';
    document.head.appendChild(link);
  }

  function removeInlineModelFlow() {
    // The mechanism explainer belongs in the top-bar model inspector, not in
    // the city viewport. Removing it restores the original large 3D map row.
    $('cwModelFlow')?.remove();
  }

  function insertCityModelButton() {
    const topStatus = q('.top-status');
    if (!topStatus || $('cwCityModelButton')) return;

    const button = document.createElement('button');
    button.id = 'cwCityModelButton';
    button.type = 'button';
    button.className = 'cw-city-model-button';
    button.setAttribute('aria-controls', 'cwCityModelModal');
    button.setAttribute('aria-expanded', 'false');
    button.innerHTML = `
      <span class="cw-city-model-pulse" aria-hidden="true"></span>
      <span class="cw-city-model-copy">
        <span class="micro-label">CITY MODEL</span>
        <strong>SAM-WM · OPEN MODEL</strong>
      </span>`;

    topStatus.insertBefore(button, topStatus.firstElementChild);
  }

  function insertCityModelModal() {
    if ($('cwCityModelModal')) return;

    const modal = document.createElement('div');
    modal.id = 'cwCityModelModal';
    modal.className = 'cw-city-model-modal';
    modal.hidden = true;
    modal.innerHTML = `
      <div class="cw-city-model-backdrop" data-close-city-model></div>
      <section
        class="cw-city-model-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cwCityModelTitle"
      >
        <header class="cw-city-model-head">
          <div>
            <div class="product-kicker">SAM-WM · CITY WORLD MODEL</div>
            <h2 id="cwCityModelTitle">HOW THIS CITY FORECAST IS BUILT</h2>
            <p>
              SAM-WM carries the recent city state forward through local spatial
              interactions and bounded thermal mechanisms, then rolls that state
              from +1 h to +6 h.
            </p>
          </div>
          <button id="cwCityModelClose" class="cw-city-model-close" type="button" aria-label="Close city model">×</button>
        </header>

        <div class="cw-city-model-current" id="cwCityModelCurrent">
          Current run · loading model context…
        </div>

        <div class="cw-city-model-flow" aria-label="SAM-WM city-model pipeline">
          <div class="cw-city-model-node">
            <b>48 H</b>
            <span>measured city history</span>
          </div>
          <i>→</i>
          <div class="cw-city-model-node">
            <b>36</b>
            <span>provider-grid tiles</span>
          </div>
          <i>→</i>
          <div class="cw-city-model-node emphasis">
            <b>LOCAL</b>
            <span>sparse city graph</span>
          </div>
          <i>→</i>
          <div class="cw-city-model-node emphasis">
            <b>ROUTE</b>
            <span>thermal mechanisms</span>
          </div>
          <i>→</i>
          <div class="cw-city-model-node">
            <b>+1…+6 H</b>
            <span>recursive future rollout</span>
          </div>
        </div>

        <div class="cw-city-model-mechanisms">
          <article>
            <strong>Conservative exchange</strong>
            <span>Neighbouring tiles exchange heat through antisymmetric pairwise flux, so the exchange term itself does not create net heat.</span>
          </article>
          <article>
            <strong>Bounded source / sink</strong>
            <span>Local unresolved heating and cooling forcing is constrained at every forecast step.</span>
          </article>
          <article>
            <strong>Bounded residual</strong>
            <span>A learned correction captures remaining dynamics but its influence is deliberately limited.</span>
          </article>
          <article>
            <strong>Wind transport</strong>
            <span>Conservative upwind transport is used only when wind is supplied. It is off for this recorded provider rollout because wind is unavailable.</span>
          </article>
          <article>
            <strong>Daily + seasonal clock</strong>
            <span>Explicit time features let the model represent diurnal and annual thermal cycles.</span>
          </article>
          <article>
            <strong>Recurrent city memory</strong>
            <span>The internal state is updated repeatedly so each predicted hour becomes context for the next forecast hour.</span>
          </article>
        </div>

        <div class="cw-city-model-footer">
          <strong>What to look for in the main view</strong>
          <span>
            Return to the 3D city, move the timeline through +1…+6 h, then inspect
            Persistent Heat Priority to see which locations remain warm across the rollout.
          </span>
        </div>
      </section>`;

    document.body.appendChild(modal);
  }

  async function updateCurrentRun() {
    const node = $('cwCityModelCurrent');
    if (!node) return;

    try {
      const response = await fetch('/api/product-status');
      if (!response.ok) return;
      const state = await response.json();
      const seed = state.selected_seed ?? '—';
      const frames = state.recorded_real_frames ?? '—';
      node.textContent = `Current run · seed ${seed} · ${frames} recorded hourly fields · latest 48 h used as forecast context · 36 spatial tiles`;
    } catch {
      node.textContent = 'Current run · frozen SAM-WM · 48 h context · 36 spatial tiles · +1…+6 h rollout';
    }
  }

  function openCityModel() {
    const modal = $('cwCityModelModal');
    const button = $('cwCityModelButton');
    if (!modal || !button) return;

    modal.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    document.body.classList.add('cw-city-model-open');
    button.classList.remove('cw-guide-focus');
    updateCurrentRun();
    $('cwCityModelClose')?.focus();
  }

  function closeCityModel() {
    const modal = $('cwCityModelModal');
    const button = $('cwCityModelButton');
    if (!modal || !button) return;

    modal.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    document.body.classList.remove('cw-city-model-open');
    button.focus();
  }

  function wireCityModel() {
    $('cwCityModelButton')?.addEventListener('click', openCityModel);
    $('cwCityModelClose')?.addEventListener('click', closeCityModel);
    q('[data-close-city-model]')?.addEventListener('click', closeCityModel);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !$('cwCityModelModal')?.hidden) {
        closeCityModel();
      }
    });
  }

  function predictedMode() {
    return /FUTURE|FORECAST/i.test($('worldStatus')?.textContent || '');
  }

  function explainTimeline() {
    const truth = $('timelineTruth');
    if (!truth) return;

    const nextText = predictedMode()
      ? 'PLAYBACK · Press ▶ or drag the slider to inspect SAM-WM from +1 h to +6 h. This changes the displayed forecast hour; it does not generate new data.'
      : 'PLAYBACK · Press ▶ or drag the slider to inspect the 65 recorded hourly city fields. Each stop is one stored provider observation.';

    // Important: write only when the content actually changes. This keeps the
    // enhancement idempotent and avoids MutationObserver feedback loops.
    if (truth.textContent !== nextText) {
      truth.textContent = nextText;
    }
  }

  function decorateGuideStep(event) {
    const panel = $('cwGuideStep');
    if (!panel) return;

    $('cwCityModelButton')?.classList.remove('cw-guide-focus');
    q('.timeline')?.classList.remove('cw-guide-timeline');

    const index = Number(event.detail?.index);
    if (index === 0) {
      q('.timeline')?.classList.add('cw-guide-timeline');
      const note = document.createElement('span');
      note.className = 'cw-guide-extra';
      note.textContent = 'Use ▶ or drag the timeline to inspect the full recorded hourly sequence before moving to the forecast.';
      panel.appendChild(note);
    }

    if (index === 1) {
      $('cwCityModelButton')?.classList.add('cw-guide-focus');
      q('.timeline')?.classList.add('cw-guide-timeline');
      const note = document.createElement('span');
      note.className = 'cw-guide-extra';
      note.innerHTML = 'After the forecast appears, use the timeline for +1…+6 h and click <strong>CITY MODEL · SAM-WM</strong> in the top bar to inspect how the forecast is produced.';
      panel.appendChild(note);
    }
  }

  function observeTimelineState() {
    const world = $('worldStatus');

    // Observe the mode source only. Never observe timelineTruth itself while
    // also writing to timelineTruth: that creates a self-triggering mutation
    // loop in browsers such as Safari and can starve the UI event loop.
    if (world) {
      new MutationObserver(explainTimeline)
        .observe(world, { childList: true, subtree: true, characterData: true });
    }

    window.setTimeout(explainTimeline, 50);
    window.setTimeout(explainTimeline, 900);
  }

  function init() {
    ensureStylesheet();
    removeInlineModelFlow();
    insertCityModelButton();
    insertCityModelModal();
    wireCityModel();
    observeTimelineState();
    document.addEventListener('coolworld:guide-step', decorateGuideStep);
  }

  document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init, { once: true })
    : init();
})();
