(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
  const PRIORITY_STYLE = 'https://tiles.openfreemap.org/styles/liberty';

  let priorityMap = null;
  let priorityPopup = null;

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
      throw new Error(String(detail));
    }
    return data;
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');
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
          <strong>${escapeHtml(label)}</strong>
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
      : 'provider grid geometry';

    return `
      <article class="hotspot-item">
        <div class="hotspot-head">
          <span class="hotspot-rank" style="--hotspot-hue:${rankHue}">#${Number(item.rank) || '—'}</span>
          <div>
            <strong>Tile ${escapeHtml(item.tile_id)}</strong>
            <span>${escapeHtml(center)}</span>
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

  function boundsOf(fc) {
    const xs = [];
    const ys = [];
    const walk = (node) => {
      if (Array.isArray(node) && typeof node[0] === 'number') {
        xs.push(Number(node[0]));
        ys.push(Number(node[1]));
      } else if (Array.isArray(node)) {
        node.forEach(walk);
      }
    };
    (fc?.features || []).forEach((feature) => walk(feature?.geometry?.coordinates));
    if (!xs.length) return null;
    return [
      [Math.min(...xs), Math.min(...ys)],
      [Math.max(...xs), Math.max(...ys)],
    ];
  }

  function firstSymbolLayer(map) {
    return (map.getStyle().layers || []).find(
      (layer) => layer.type === 'symbol' && layer.layout?.['text-field'],
    )?.id;
  }

  function ensurePriorityBuildings(map) {
    const layers = map.getStyle().layers || [];
    const already3d = layers.some(
      (layer) => layer.type === 'fill-extrusion'
        && String(layer['source-layer'] || '').toLowerCase().includes('building'),
    );
    if (already3d || map.getLayer('cw-priority-buildings')) return;

    const candidate = layers.find(
      (layer) => String(layer['source-layer'] || '').toLowerCase().includes('building')
        && layer.source,
    );
    if (!candidate) return;

    map.addLayer({
      id: 'cw-priority-buildings',
      type: 'fill-extrusion',
      source: candidate.source,
      'source-layer': candidate['source-layer'],
      minzoom: 13,
      paint: {
        'fill-extrusion-height': [
          'coalesce',
          ['to-number', ['get', 'render_height']],
          ['to-number', ['get', 'height']],
          0,
        ],
        'fill-extrusion-base': [
          'coalesce',
          ['to-number', ['get', 'render_min_height']],
          ['to-number', ['get', 'min_height']],
          0,
        ],
        'fill-extrusion-color': '#87949d',
        'fill-extrusion-opacity': 0.70,
      },
    }, firstSymbolLayer(map));
  }

  function ensurePriorityLayers(map, featureCollection) {
    const before = firstSymbolLayer(map);
    if (!map.getSource('cw-hotspot-priority')) {
      map.addSource('cw-hotspot-priority', {
        type: 'geojson',
        data: featureCollection,
      });
      map.addLayer({
        id: 'cw-hotspot-priority-fill',
        type: 'fill',
        source: 'cw-hotspot-priority',
        paint: {
          'fill-color': [
            'interpolate', ['linear'], ['to-number', ['get', 'cw_priority_normalized']],
            0.0, '#f6df6e',
            0.45, '#f0a34d',
            0.75, '#df6839',
            1.0, '#b02f2d',
          ],
          'fill-opacity': 0.72,
        },
      }, before);
      map.addLayer({
        id: 'cw-hotspot-priority-line',
        type: 'line',
        source: 'cw-hotspot-priority',
        paint: {
          'line-color': '#52221f',
          'line-width': 1.0,
          'line-opacity': 0.85,
        },
      }, before);
    } else {
      map.getSource('cw-hotspot-priority').setData(featureCollection);
    }
    ensurePriorityBuildings(map);
  }

  function buildPriorityField(geometry, data) {
    const selected = new Map((data.hotspots || []).map((item) => [String(item.tile_id), item]));
    const selectedCount = Math.max(1, selected.size);
    const features = (geometry?.features || [])
      .filter((feature) => selected.has(String(feature?.properties?.tile_id ?? feature?.id ?? '')))
      .map((feature) => {
        const id = String(feature?.properties?.tile_id ?? feature?.id ?? '');
        const item = selected.get(id);
        const rank = Math.max(1, Number(item.rank) || 1);
        const normalized = selectedCount <= 1
          ? 1
          : 1 - ((rank - 1) / (selectedCount - 1));
        return {
          ...feature,
          properties: {
            ...(feature.properties || {}),
            tile_id: id,
            cw_priority_normalized: normalized,
            cw_priority_rank: rank,
            cw_current_temperature_c: item.current_temperature_c,
            cw_forecast_6h_c: item.forecast_6h_c,
            cw_forecast_max_c: item.forecast_max_c,
            cw_hotspot_persistence: item.top_fraction_persistence,
          },
        };
      });
    return { type: 'FeatureCollection', features };
  }

  async function renderHotspotMap(data) {
    const host = $('hotspotMap');
    if (!host || !window.maplibregl) return;

    const timeline = await getJson('/api/evidence/timeline?limit=1');
    const geometry = timeline.frames?.[timeline.frames.length - 1]?.map_data;
    const field = buildPriorityField(geometry, data);
    if (!field.features.length) {
      host.classList.add('empty');
      return;
    }

    host.classList.remove('empty');

    if (!priorityMap) {
      priorityMap = new maplibregl.Map({
        container: 'hotspotMap',
        style: PRIORITY_STYLE,
        center: [-121.8863, 37.3382],
        zoom: 13.8,
        pitch: 58,
        bearing: -24,
        antialias: true,
        attributionControl: false,
      });
      priorityMap.addControl(
        new maplibregl.NavigationControl({ visualizePitch: true }),
        'top-right',
      );
      priorityPopup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false,
        offset: 10,
      });

      priorityMap.on('load', () => {
        ensurePriorityLayers(priorityMap, field);
        const bounds = boundsOf(field);
        if (bounds) {
          priorityMap.fitBounds(bounds, {
            padding: 32,
            maxZoom: 15.5,
            duration: 0,
          });
        }
      });

      priorityMap.on('mousemove', 'cw-hotspot-priority-fill', (event) => {
        const feature = event.features?.[0];
        if (!feature || !priorityPopup) return;
        const p = feature.properties || {};
        const wrapper = document.createElement('div');
        wrapper.className = 'priority-popup';
        const title = document.createElement('strong');
        title.textContent = `Priority #${p.cw_priority_rank} · Tile ${p.tile_id}`;
        const details = document.createElement('span');
        details.textContent = `Now ${fmt(p.cw_current_temperature_c)} °C · +6 h ${fmt(p.cw_forecast_6h_c)} °C · persistence ${percent(p.cw_hotspot_persistence, 0)}`;
        wrapper.append(title, details);
        priorityPopup.setLngLat(event.lngLat).setDOMContent(wrapper).addTo(priorityMap);
        priorityMap.getCanvas().style.cursor = 'pointer';
      });

      priorityMap.on('mouseleave', 'cw-hotspot-priority-fill', () => {
        priorityPopup?.remove();
        priorityMap.getCanvas().style.cursor = '';
      });
    } else if (priorityMap.loaded()) {
      ensurePriorityLayers(priorityMap, field);
      const bounds = boundsOf(field);
      if (bounds) {
        priorityMap.fitBounds(bounds, {
          padding: 32,
          maxZoom: 15.5,
          duration: 450,
        });
      }
      priorityMap.resize();
    }
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
          <div><span>MODEL</span><strong>${escapeHtml(data.model)}</strong></div>
          <div><span>SEED</span><strong>${escapeHtml(data.selected_seed ?? '—')}</strong></div>
          <div><span>REAL FRAMES</span><strong>${escapeHtml(data.recorded_real_frames ?? '—')}</strong></div>
        </div>
        ${benchmarkLine('Freiburg · final ID', benchmarks.freiburg_id)}
        ${benchmarkLine('Novi Sad · zero-shot OOD-1', benchmarks.novi_sad_ood)}
        ${benchmarkLine('Turku · zero-shot OOD-2', benchmarks.turku_fairurbtemp_ood)}
        <div class="replay-gate ${replay.status === 'PASS' ? 'pass' : 'fail'}">
          <strong>FortyGuard operational replay: ${escapeHtml(replay.status || 'UNKNOWN')}</strong>
          <span>coverage ${percent(replay.conformal_coverage, 2)} vs fixed ${percent(replay.minimum_required_coverage, 2)} gate</span>
          <span>MAE/radius ${fmt(replay.mae_to_radius_ratio, 3)} vs max ${fmt(replay.maximum_allowed_mae_to_radius_ratio, 3)}</span>
        </div>
        <div class="claim-boundary">
          Forecasting evidence is real and reproducible. Cooling-effect magnitude remains intentionally unclaimed without treated/control intervention evidence.
        </div>
      `;
    } catch (error) {
      host.innerHTML = `<div class="product-error">Evidence summary unavailable: ${escapeHtml(error.message)}</div>`;
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
      status.textContent = `${data.selected_count} of ${data.tile_count} tiles prioritized · relative future heat · non-causal decision support`;
      host.innerHTML = data.hotspots
        .map((item) => hotspotCard(item, data.conformal_radius_c))
        .join('');

      const boundary = $('hotspotBoundary');
      if (boundary) boundary.textContent = data.claim_boundary;

      await renderHotspotMap(data);
    } catch (error) {
      status.textContent = 'Hotspot plan unavailable';
      host.innerHTML = `<div class="product-error">${escapeHtml(error.message)}</div>`;
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
      window.setTimeout(() => priorityMap?.resize(), 400);
    });

    $('guideEvidence')?.addEventListener('click', async () => {
      await refreshEvidence();
      scrollTo('evidenceCardAnchor');
    });

    $('refreshHotspots')?.addEventListener('click', refreshHotspots);
    $('refreshEvidence')?.addEventListener('click', refreshEvidence);
    $('targetFraction')?.addEventListener('change', () => {
      const node = $('hotspotStatus');
      if (node) node.textContent = 'Target fraction changed · press Refresh to recompute priority view.';
    });

    try {
      await refreshProductStatus();
    } catch (error) {
      const node = $('productStatusError');
      if (node) {
        node.classList.remove('hidden');
        node.textContent = `Status endpoint unavailable: ${error.message}`;
      }
    }

    await refreshEvidence();
  }

  window.addEventListener('load', () => {
    window.setTimeout(init, 900);
  });
})();
