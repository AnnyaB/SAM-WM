(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  async function getJson(url, options = {}) {
    const response = await fetch(url, options);
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    if (!response.ok) {
      const detail = data?.detail || `${response.status} ${response.statusText}`;
      throw new Error(detail);
    }
    return data;
  }

  function fmt(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : '—';
  }

  function percent(value, digits = 1) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(100 * number).toFixed(digits)}%` : '—';
  }

  function setStatusPill(id, ready, text) {
    const node = $(id);
    if (!node) return;
    node.textContent = text;
    node.classList.toggle('ok', Boolean(ready));
    node.classList.toggle('warn', !ready);
  }

  function benchmarkLine(label, item) {
    const mae = item?.mae_c;
    const coverage = item?.conformal_coverage;
    return `
      <div class="evidence-row">
        <div>
          <strong>${label}</strong>
          <span>MAE ${fmt(mae?.mean, 3)} ± ${fmt(mae?.std, 3)} °C</span>
        </div>
        <span class="evidence-chip">coverage ${percent(coverage?.mean, 1)}</span>
      </div>
    `;
  }

  function hotspotCard(item, radius) {
    const persistence = Math.max(0, Math.min(1, Number(item.top_fraction_persistence) || 0));
    const percentileValue = Math.max(0, Math.min(1, Number(item.future_hotspot_percentile) || 0));
    const rankHue = 210 - (170 * percentileValue);
    const center = item.center
      ? `${fmt(item.center.lat, 5)}, ${fmt(item.center.lon, 5)}`
      : 'grid location available in provider geometry';

    return `
      <article class="hotspot-item">
        <div class="hotspot-head">
          <span class="hotspot-rank" style="--hotspot-hue:${rankHue}">#${item.rank}</span>
          <div>
            <strong>Tile ${item.tile_id}</strong>
            <span>${center}</span>
          </div>
        </div>
        <div class="hotspot-metrics">
          <span>Now <strong>${fmt(item.current_temperature_c)} °C</strong></span>
          <span>+6 h <strong>${fmt(item.forecast_6h_c)} °C</strong></span>
          <span>Future max <strong>${fmt(item.forecast_max_c)} °C</strong></span>
          <span>± radius <strong>${fmt(radius, 3)} °C</strong></span>
        </div>
        <div class="hotspot-bar" aria-label="future hotspot persistence">
          <span style="width:${(100 * persistence).toFixed(1)}%"></span>
        </div>
        <div class="hotspot-foot">
          Persistent top-zone: ${percent(persistence, 0)} · candidate engineering review: trees / shade / reflective surface · causal ΔT not estimated
        </div>
      </article>
    `;
  }

  async function refreshProductStatus() {
    const state = await getJson('/api/product-status');

    setStatusPill(
      'statusRealEvidence',
      state.real_provider_evidence_ready,
      state.real_provider_evidence_ready
        ? `${state.recorded_real_frames} real provider frames`
        : 'real provider evidence missing',
    );
    setStatusPill(
      'statusModelBundle',
      state.model_bundle_promoted,
      state.model_bundle_promoted
        ? `frozen seed ${state.selected_seed ?? '—'} promoted`
        : state.promotion_reason || 'promotion not verified',
    );
    setStatusPill(
      'statusResearchForecast',
      state.research_forecast_ready,
      state.research_forecast_ready
        ? 'research forecast ready'
        : 'research forecast unavailable',
    );
    setStatusPill(
      'statusOperational',
      state.operational_certified,
      state.operational_certified
        ? 'operational replay certified'
        : 'operational replay not certified',
    );
    setStatusPill(
      'statusCausal',
      state.causal_action_ready,
      state.causal_action_ready
        ? 'causal action evidence ready'
        : 'causal action evidence required',
    );

    const liveNote = $('liveApiState');
    if (liveNote) {
      if (state.live_provider_api_enabled && state.fortyguard_key_configured) {
        liveNote.textContent = 'LIVE API ENABLED · requests may consume provider allocation.';
        liveNote.className = 'live-api-state warn-text';
      } else {
        liveNote.textContent = 'RECORDED-EVIDENCE MODE · zero new provider requests.';
        liveNote.className = 'live-api-state safe-text';
      }
    }

    const runHeatmap = $('runHeatmap');
    if (runHeatmap && !state.live_provider_api_enabled) {
      runHeatmap.disabled = true;
      runHeatmap.title = 'Public-safe default: live provider calls are disabled. Set COOLWORLD_LIVE_API_ENABLED=1 server-side to enable them.';
      runHeatmap.textContent = 'Live provider request disabled';
    }

    return state;
  }

  async function refreshEvidence() {
    const host = $('evidenceSummary');
    if (!host) return;
    host.innerHTML = '<div class="product-loading">Loading immutable evidence…</div>';

    try {
      const data = await getJson('/api/evidence-summary');
      const replay = data.provider_replay || {};
      const benchmarks = data.benchmarks || {};
      host.innerHTML = `
        <div class="evidence-hero">
          <div><span>MODEL</span><strong>${data.model}</strong></div>
          <div><span>SEED</span><strong>${data.selected_seed ?? '—'}</strong></div>
          <div><span>REAL FRAMES</span><strong>${data.recorded_real_frames ?? '—'}</strong></div>
        </div>
        ${benchmarkLine('Freiburg · final ID', benchmarks.freiburg_id)}
        ${benchmarkLine('Novi Sad · zero-shot OOD-1', benchmarks.novi_sad_ood)}
        ${benchmarkLine('Turku · zero-shot OOD-2', benchmarks.turku_fairurbtemp_ood)}
        <div class="replay-gate ${replay.status === 'PASS' ? 'pass' : 'fail'}">
          <strong>FortyGuard operational replay: ${replay.status || 'UNKNOWN'}</strong>
          <span>coverage ${percent(replay.conformal_coverage, 2)} vs fixed ${percent(replay.minimum_required_coverage, 2)} gate</span>
          <span>MAE/radius ${fmt(replay.mae_to_radius_ratio, 3)} vs max ${fmt(replay.maximum_allowed_mae_to_radius_ratio, 3)}</span>
        </div>
        <div class="claim-boundary">
          Forecasting evidence is real and reproducible. Cooling-effect magnitude remains intentionally unclaimed without treated/control intervention evidence.
        </div>
      `;
    } catch (error) {
      host.innerHTML = `<div class="product-error">Evidence summary unavailable: ${error.message}</div>`;
    }
  }

  async function refreshHotspots() {
    const host = $('hotspotList');
    const status = $('hotspotStatus');
    if (!host || !status) return;

    status.textContent = 'Running frozen SAM-WM hotspot ranking…';
    host.innerHTML = '<div class="product-loading">Forecast-derived priority analysis…</div>';

    try {
      const fraction = Number($('targetFraction')?.value || 0.2);
      const data = await getJson(`/api/hotspots?fraction=${encodeURIComponent(fraction)}`);
      status.textContent = `${data.selected_count} of ${data.tile_count} tiles prioritized · non-causal decision support`;
      host.innerHTML = data.hotspots
        .map((item) => hotspotCard(item, data.conformal_radius_c))
        .join('');

      const boundary = $('hotspotBoundary');
      if (boundary) boundary.textContent = data.claim_boundary;
    } catch (error) {
      status.textContent = 'Hotspot plan unavailable';
      host.innerHTML = `<div class="product-error">${error.message}</div>`;
    }
  }

  function scrollTo(id) {
    $(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function ensureObserved() {
    const observed = document.querySelector('[data-mode="observed"]');
    observed?.click();
    const frameLabel = $('frameLabel')?.textContent || '';
    if (/^0\s+frames?$/.test(frameLabel.trim())) {
      $('loadTimeline')?.click();
      await sleep(700);
    }
  }

  async function ensureForecast() {
    await ensureObserved();
    $('predictedTab')?.click();
    await sleep(900);
  }

  async function init() {
    const predicted = $('predictedTab');
    if (predicted) predicted.textContent = 'SAM-WM FORECAST';
    const replay = $('replayTab');
    if (replay) replay.textContent = 'INTERVENTION EVIDENCE';
    const predict = $('predict');
    if (predict) predict.textContent = 'Check supported action effect';

    $('guideObserve')?.addEventListener('click', async () => {
      await ensureObserved();
      scrollTo('measuredCardAnchor');
    });

    $('guideForecast')?.addEventListener('click', async () => {
      await ensureForecast();
      scrollTo('forecastAnalyticsAnchor');
    });

    $('guideHotspots')?.addEventListener('click', async () => {
      await ensureForecast();
      await refreshHotspots();
      scrollTo('hotspotCardAnchor');
    });

    $('guideEvidence')?.addEventListener('click', async () => {
      await refreshEvidence();
      scrollTo('evidenceCardAnchor');
    });

    $('refreshHotspots')?.addEventListener('click', refreshHotspots);
    $('refreshEvidence')?.addEventListener('click', refreshEvidence);

    try {
      await refreshProductStatus();
    } catch (error) {
      const node = $('productStatusError');
      if (node) node.textContent = `Status endpoint unavailable: ${error.message}`;
    }

    await refreshEvidence();
  }

  window.addEventListener('load', () => {
    window.setTimeout(init, 900);
  });
})();
