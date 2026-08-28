/* SAM-WM v0.6 browser renderer.
 * Reality contract:
 * - Basemap / building geometry: real map tiles only.
 * - Observed thermal values: real FortyGuard evidence only.
 * - Predicted thermal values: explicit model prediction only.
 * - Camera motion / between-frame interpolation: visualization only.
 * - No invented intervention structure height and no hard-coded cooling effect.
 */
(() => {
  'use strict';

  if (!window.maplibregl) {
    throw new Error('MapLibre did not load; refusing to fabricate a map');
  }

  const PRIMARY_STYLE = 'https://tiles.openfreemap.org/styles/liberty';
  const FALLBACK_STYLE = 'https://demotiles.maplibre.org/style.json';
  const DEFAULT_VIEW = {
    center: [-121.8863, 37.3382],
    zoom: 14.2,
    pitch: 58,
    bearing: -24,
  };

  const state = {
    mode: 'observed',
    observedFrames: [],
    baselineFrames: [],
    candidateFrames: [],
    geometry: null,
    gridSignature: null,
    observedDomain: null,
    predictedDomain: null,
    prediction: null,
    playing: false,
    playhead: 0,
    timelineRaf: null,
    aoi: [],
    drawing: false,
    modelReady: false,
    researchPreviewReady: false,
    selectedIds: [],
    orbiting: false,
    orbitRaf: null,
    buildingLayers: new Map(),
    activeStyle: PRIMARY_STYLE,
    cameraReasserted: new Set(),
  };

  const $ = (id) => document.getElementById(id);
  const svgNS = 'http://www.w3.org/2000/svg';

  const log = (message) => {
    const stamp = new Date().toISOString().slice(11, 19);
    const consoleNode = $('console');
    consoleNode.textContent += `[${stamp}] ${message}\n`;
    consoleNode.scrollTop = consoleNode.scrollHeight;
  };

  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const tileId = (feature) => String(feature?.properties?.tile_id ?? feature?.id ?? '');
  const observedTempOf = (feature) => Number(
    feature?.properties?.cw_observed_temperature_c ?? feature?.properties?.average_temperature,
  );
  const modelTempOf = (feature) => Number(feature?.properties?.cw_model_temperature_c);
  const renderTempOf = (feature) => {
    const rendered = Number(feature?.properties?.cw_render_temperature_c);
    if (Number.isFinite(rendered)) return rendered;
    const model = modelTempOf(feature);
    if (Number.isFinite(model)) return model;
    return observedTempOf(feature);
  };

  const finiteValues = (fc, accessor = renderTempOf) => (fc?.features || [])
    .map(accessor)
    .filter(Number.isFinite);

  const percentile = (values, p) => {
    if (!values.length) return NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const index = (sorted.length - 1) * p;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) return sorted[lower];
    const weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  };

  const temperatureDomain = (frames, accessor) => {
    const values = [];
    for (const frame of frames) {
      values.push(...finiteValues(frame.map_data, accessor));
    }
    if (!values.length) return null;
    let lo = Math.min(...values);
    let hi = Math.max(...values);
    if (Math.abs(hi - lo) < 1e-6) {
      lo -= 0.5;
      hi += 0.5;
    }
    return [lo, hi];
  };

  const mergedPredictionDomain = () => {
    const all = [...state.baselineFrames, ...state.candidateFrames];
    return temperatureDomain(all, modelTempOf);
  };

  function firstSymbolLayer(map) {
    return (map.getStyle().layers || []).find(
      (layer) => layer.type === 'symbol' && layer.layout?.['text-field'],
    )?.id;
  }

  function addBuildingExtrusion(map) {
    const layers = map.getStyle().layers || [];
    const existing = layers.filter(
      (layer) => layer.type === 'fill-extrusion'
        && String(layer['source-layer'] || '').toLowerCase().includes('building'),
    );
    if (existing.length) {
      state.buildingLayers.set(map.getContainer().id, existing.map((layer) => layer.id));
      log(`${map.getContainer().id}: using basemap-provided 3D building layer(s).`);
      if (map.getContainer().id === 'mapA') $('buildingStatus').textContent = '3D BUILDINGS: REAL BASEMAP LAYER';
      return;
    }

    const candidate = layers.find((layer) => (
      String(layer['source-layer'] || '').toLowerCase().includes('building')
      && layer.source
    ));
    if (!candidate || map.getLayer('cw-buildings-3d')) {
      state.buildingLayers.set(map.getContainer().id, []);
      log(`${map.getContainer().id}: no explicit building height source available; no invented height.`);
      if (map.getContainer().id === 'mapA') $('buildingStatus').textContent = '3D BUILDINGS: HEIGHT DATA UNAVAILABLE IN THIS STYLE';
      return;
    }

    const height = [
      'coalesce',
      ['to-number', ['get', 'render_height']],
      ['to-number', ['get', 'height']],
      0,
    ];
    const base = [
      'coalesce',
      ['to-number', ['get', 'render_min_height']],
      ['to-number', ['get', 'min_height']],
      0,
    ];

    map.addLayer({
      id: 'cw-buildings-3d',
      type: 'fill-extrusion',
      source: candidate.source,
      'source-layer': candidate['source-layer'],
      minzoom: 13,
      paint: {
        'fill-extrusion-height': height,
        'fill-extrusion-base': base,
        'fill-extrusion-opacity': 0.78,
        'fill-extrusion-color': '#82929c',
      },
    }, firstSymbolLayer(map));
    state.buildingLayers.set(map.getContainer().id, ['cw-buildings-3d']);
    log(`${map.getContainer().id}: explicit-height 3D building extrusion enabled.`);
    if (map.getContainer().id === 'mapA') $('buildingStatus').textContent = '3D BUILDINGS: EXPLICIT REAL HEIGHTS';
  }

  function ensureWorldSources(map) {
    const before = firstSymbolLayer(map);
    if (!map.getSource('cw-thermal')) {
      map.addSource('cw-thermal', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'cw-thermal-fill',
        type: 'fill',
        source: 'cw-thermal',
        paint: {
          'fill-color': '#2a89a4',
          'fill-opacity': 0.62,
        },
      }, before);
      map.addLayer({
        id: 'cw-thermal-line',
        type: 'line',
        source: 'cw-thermal',
        paint: {
          'line-color': ['case', ['boolean', ['get', 'cw_selected'], false], '#7ce0a7', '#15232c'],
          'line-width': ['case', ['boolean', ['get', 'cw_selected'], false], 2.2, 0.35],
          'line-opacity': 0.72,
        },
      }, before);
    }

    if (!map.getSource('cw-action')) {
      map.addSource('cw-action', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'cw-action-fill',
        type: 'fill',
        source: 'cw-action',
        paint: {
          'fill-color': ['match', ['get', 'cw_action_kind'],
            'shade', '#32cda4',
            'tree_canopy', '#36bd64',
            'reflective_pavement', '#d7e2e8',
            '#7ce0a7'],
          'fill-opacity': 0.36,
        },
      }, before);
      map.addLayer({
        id: 'cw-action-line',
        type: 'line',
        source: 'cw-action',
        paint: {
          'line-color': '#b9ffda',
          'line-width': 2.2,
          'line-opacity': 0.9,
        },
      }, before);
    }

    if (!map.getSource('cw-aoi')) {
      map.addSource('cw-aoi', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'cw-aoi-fill',
        type: 'fill',
        source: 'cw-aoi',
        paint: { 'fill-color': '#68bee9', 'fill-opacity': 0.09 },
      }, before);
      map.addLayer({
        id: 'cw-aoi-line',
        type: 'line',
        source: 'cw-aoi',
        paint: { 'line-color': '#68bee9', 'line-width': 2.2, 'line-dasharray': [2, 1] },
      }, before);
    }
  }

  function reassertDefaultCamera(map, reason) {
    if (state.geometry) return;
    const key = `${map.getContainer().id}:${reason}`;
    if (state.cameraReasserted.has(key)) return;
    state.cameraReasserted.add(key);
    map.jumpTo(DEFAULT_VIEW);
    log(`${map.getContainer().id}: camera focused on San José base view (${reason}).`);
  }

  function makeMap(container) {
    const map = new maplibregl.Map({
      container,
      style: PRIMARY_STYLE,
      ...DEFAULT_VIEW,
      antialias: true,
      attributionControl: true,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');

    let loaded = false;
    let fallbackUsed = false;
    const fallback = (reason) => {
      if (loaded || fallbackUsed) return;
      fallbackUsed = true;
      state.activeStyle = FALLBACK_STYLE;
      log(`${container}: primary basemap unavailable (${reason}); trying MapLibre fallback style.`);
      map.setStyle(FALLBACK_STYLE);
    };

    const watchdog = window.setTimeout(() => fallback('load timeout'), 8000);
    map.on('error', (event) => {
      const message = event?.error?.message || 'map error';
      if (!loaded && !fallbackUsed) {
        fallback(message);
      } else {
        log(`${container}: map warning: ${message}`);
      }
    });

    map.on('load', () => {
      loaded = true;
      window.clearTimeout(watchdog);
      ensureWorldSources(map);
      addBuildingExtrusion(map);
      reassertDefaultCamera(map, 'load');
      log(`${container}: real basemap ready.`);
      map.resize();
    });

    map.on('styledata', () => {
      if (!map.isStyleLoaded()) return;
      try {
        ensureWorldSources(map);
        addBuildingExtrusion(map);
        applyLayerVisibility(map);
        reassertDefaultCamera(map, fallbackUsed ? 'fallback-style' : 'styledata');
        rerenderForMap(map);
      } catch (error) {
        log(`${container}: style synchronization warning: ${error.message}`);
      }
    });

    return { map };
  }

  const A = makeMap('mapA');
  const B0 = makeMap('mapBaseline');
  const B1 = makeMap('mapCandidate');

  let syncing = false;
  const syncMaps = (source, destination) => {
    source.on('move', () => {
      if (syncing) return;
      syncing = true;
      const center = source.getCenter();
      destination.jumpTo({
        center,
        zoom: source.getZoom(),
        bearing: source.getBearing(),
        pitch: source.getPitch(),
      });
      syncing = false;
    });
  };
  syncMaps(B0.map, B1.map);
  syncMaps(B1.map, B0.map);

  function colorStops(domain) {
    const [lo, hi] = domain;
    const q1 = lo + (hi - lo) * 0.2;
    const q2 = lo + (hi - lo) * 0.4;
    const q3 = lo + (hi - lo) * 0.6;
    const q4 = lo + (hi - lo) * 0.8;
    return [
      'interpolate', ['linear'], ['to-number', ['get', 'cw_render_temperature_c']],
      lo, '#235789',
      q1, '#2a89a4',
      q2, '#6db78d',
      q3, '#f6df6e',
      q4, '#e67c41',
      hi, '#b02f2d',
    ];
  }

  function prepareRenderedField(fc, selectedIds = []) {
    const selected = new Set(selectedIds.map(String));
    return {
      type: 'FeatureCollection',
      features: (fc?.features || []).map((feature) => ({
        ...feature,
        properties: {
          ...(feature.properties || {}),
          cw_render_temperature_c: renderTempOf(feature),
          cw_selected: selected.has(tileId(feature)),
        },
      })),
    };
  }

  function renderField(target, fc, domain, selectedIds = []) {
    const map = target.map;
    if (!map.loaded() || !map.getSource('cw-thermal') || !fc || !domain) return;
    const prepared = prepareRenderedField(fc, selectedIds);
    map.getSource('cw-thermal').setData(prepared);
    map.setPaintProperty('cw-thermal-fill', 'fill-color', colorStops(domain));
    map.setPaintProperty('cw-thermal-fill', 'fill-opacity', 0.62);
  }

  function clearField(target) {
    const map = target.map;
    if (map.loaded() && map.getSource('cw-thermal')) {
      map.getSource('cw-thermal').setData({ type: 'FeatureCollection', features: [] });
    }
  }

  function renderActionFootprint(target, fc, actionKind, selectedIds) {
    const map = target.map;
    if (!map.loaded() || !map.getSource('cw-action')) return;
    const selected = new Set(selectedIds.map(String));
    const features = (fc?.features || [])
      .filter((feature) => selected.has(tileId(feature)))
      .map((feature) => ({
        ...feature,
        properties: {
          ...(feature.properties || {}),
          cw_action_kind: actionKind,
        },
      }));
    map.getSource('cw-action').setData({ type: 'FeatureCollection', features });
  }

  function clearAction(target) {
    const map = target.map;
    if (map.loaded() && map.getSource('cw-action')) {
      map.getSource('cw-action').setData({ type: 'FeatureCollection', features: [] });
    }
  }

  function applyLayerVisibility(map) {
    const toggle = (id, visible) => {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none');
    };
    const buildingsVisible = $('toggleBuildings')?.checked ?? true;
    const thermalVisible = $('toggleThermal')?.checked ?? true;
    const actionVisible = $('toggleAction')?.checked ?? true;
    for (const id of state.buildingLayers.get(map.getContainer().id) || []) toggle(id, buildingsVisible);
    toggle('cw-thermal-fill', thermalVisible);
    toggle('cw-thermal-line', thermalVisible);
    toggle('cw-action-fill', actionVisible);
    toggle('cw-action-line', actionVisible);
  }

  function rerenderForMap(map) {
    if (state.mode === 'observed' && state.geometry && state.observedDomain) {
      if (map === A.map) updateAt(state.playhead);
    }
    if (state.mode === 'predicted' && state.predictedDomain && state.baselineFrames.length) {
      updateAt(state.playhead);
    }
    updateAoiLayer();
  }

  function boundsOf(fc) {
    const xs = [];
    const ys = [];
    const walk = (node) => {
      if (Array.isArray(node) && typeof node[0] === 'number') {
        xs.push(node[0]);
        ys.push(node[1]);
      } else if (Array.isArray(node)) {
        node.forEach(walk);
      }
    };
    (fc?.features || []).forEach((feature) => walk(feature.geometry?.coordinates));
    return xs.length
      ? [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]]
      : null;
  }

  function fitAll(fc) {
    const bounds = boundsOf(fc);
    if (!bounds) return;
    [A.map, B0.map, B1.map].forEach((map) => {
      map.fitBounds(bounds, { padding: 70, maxZoom: 16.2, duration: 850 });
    });
  }

  function frameFromModelTemps(geometry, tileIds, temperatures) {
    const tempMap = new Map(tileIds.map((id, index) => [String(id), Number(temperatures[index])]));
    return {
      type: 'FeatureCollection',
      features: (geometry?.features || []).map((feature) => {
        const temperature = tempMap.get(tileId(feature));
        const properties = {
          ...(feature.properties || {}),
          cw_mode: 'model_prediction',
        };
        if (Number.isFinite(temperature)) properties.cw_model_temperature_c = temperature;
        return { ...feature, properties };
      }),
    };
  }

  function interpolate(a, b, fraction, mode) {
    if (!a || !b) return a || b;
    const bMap = new Map((b.features || []).map((feature) => [tileId(feature), renderTempOf(feature)]));
    return {
      type: 'FeatureCollection',
      features: (a.features || []).map((feature) => {
        const first = renderTempOf(feature);
        const second = bMap.get(tileId(feature));
        const value = Number.isFinite(first) && Number.isFinite(second)
          ? first + (second - first) * fraction
          : first;
        return {
          ...feature,
          properties: {
            ...(feature.properties || {}),
            cw_render_temperature_c: value,
            cw_render_mode: fraction > 1e-6 ? 'visual_interpolation' : mode,
          },
        };
      }),
    };
  }

  function currentFrames() {
    return state.mode === 'observed' ? state.observedFrames : state.baselineFrames;
  }

  function updateTimelineControls() {
    const frames = currentFrames();
    const slider = $('timeSlider');
    slider.max = String(Math.max(0, frames.length - 1));
    slider.disabled = frames.length < 2;
    $('playButton').disabled = frames.length < 2;
    $('frameLabel').textContent = `${frames.length} frame${frames.length === 1 ? '' : 's'}`;
  }

  function updateAt(playhead) {
    const frames = currentFrames();
    if (!frames.length) return;
    const max = frames.length - 1;
    const value = clamp(playhead, 0, max);
    const i = Math.floor(value);
    const j = Math.min(max, i + 1);
    const fraction = value - i;

    if (state.mode === 'observed') {
      const fc = interpolate(frames[i].map_data, frames[j].map_data, fraction, 'observed');
      renderField(A, fc, state.observedDomain);
      clearAction(A);
      updateObservedAnalytics(fc, fraction > 1e-6);
      $('timeLabel').textContent = frames[i].timestamp || 'OBSERVED';
      $('timelineTruth').textContent = fraction > 1e-6
        ? 'VISUAL INTERPOLATION between real recorded frames.'
        : 'Exact recorded real frame.';
    } else {
      const baseline = interpolate(
        state.baselineFrames[i].map_data,
        state.baselineFrames[j].map_data,
        fraction,
        'model_prediction',
      );

      if (state.candidateFrames.length) {
        const candidate = interpolate(
          state.candidateFrames[i].map_data,
          state.candidateFrames[j].map_data,
          fraction,
          'model_prediction',
        );

        renderField(
          B0,
          baseline,
          state.predictedDomain,
          state.selectedIds,
        );

        renderField(
          B1,
          candidate,
          state.predictedDomain,
          state.selectedIds,
        );

        clearAction(B0);

        renderActionFootprint(
          B1,
          state.geometry,
          $('actionKind').value,
          state.selectedIds,
        );
      } else {
        // Baseline-only research forecast.
        // Render on the primary real 3D city map.
        renderField(
          A,
          baseline,
          state.predictedDomain,
          [],
        );

        clearAction(A);
      }

      $('timeLabel').textContent =
        state.baselineFrames[i].timestamp
        || `MODEL +${i + 1}`;

      $('timelineTruth').textContent = fraction > 1e-6
        ? 'VISUAL INTERPOLATION between frozen SAM-WM frames — NOT OBSERVED.'
        : 'FROZEN SAM-WM RESEARCH FORECAST — NOT OBSERVED.';
    }

    state.playhead = value;
    $('timeSlider').value = String(value);
  }

  function animateTimeline(timestamp) {
    if (!state.playing) return;
    const frames = currentFrames();
    if (frames.length < 2) {
      stopTimeline();
      return;
    }
    if (!animateTimeline.last) animateTimeline.last = timestamp;
    const dt = timestamp - animateTimeline.last;
    animateTimeline.last = timestamp;
    state.playhead += dt / 950;
    if (state.playhead >= frames.length - 1) state.playhead = 0;
    updateAt(state.playhead);
    state.timelineRaf = requestAnimationFrame(animateTimeline);
  }
  animateTimeline.last = 0;

  function stopTimeline() {
    state.playing = false;
    $('playButton').textContent = '▶';
    animateTimeline.last = 0;
    if (state.timelineRaf) cancelAnimationFrame(state.timelineRaf);
  }

  $('playButton').addEventListener('click', () => {
    if (state.playing) {
      stopTimeline();
      return;
    }
    state.playing = true;
    $('playButton').textContent = '❚❚';
    animateTimeline.last = 0;
    state.timelineRaf = requestAnimationFrame(animateTimeline);
  });

  $('timeSlider').addEventListener('input', (event) => {
    stopTimeline();
    updateAt(Number(event.target.value));
  });

  function setMode(mode) {
    stopTimeline();
    if (
      mode === 'predicted'
      && !state.researchPreviewReady
      && !state.baselineFrames.length
    ) {
      log(
        'RESEARCH_FORECAST_NOT_READY — '
        + 'a frozen checkpoint and 48 real context hours are required.',
      );
      mode = 'observed';
    }
    state.mode = mode;
    document.querySelectorAll('[data-mode]').forEach((button) => {
      button.classList.toggle('active', button.dataset.mode === mode);
    });
    const comparingCounterfactual =
      mode === 'predicted'
      && state.candidateFrames.length > 0;

    $('singleWorld').classList.toggle(
      'hidden',
      comparingCounterfactual,
    );

    $('compareWorld').classList.toggle(
      'hidden',
      !comparingCounterfactual,
    );

    if (mode === 'observed') {
      $('worldStatus').textContent = state.observedFrames.length ? 'OBSERVED REAL' : 'OBSERVED BASE MAP';
      $('modeBanner').textContent = state.observedFrames.length
        ? 'OBSERVED · REAL FORTYGUARD EVIDENCE'
        : 'BASE MAP ONLY · LOAD REAL FORTYGUARD EVIDENCE';
      if (!state.observedFrames.length) {
        $('timeLabel').textContent = 'NO RECORDED REAL TIMELINE';
      }
    } else {
      $('worldStatus').textContent =
        state.candidateFrames.length
          ? 'PREDICTED COUNTERFACTUAL'
          : 'SAM-WM RESEARCH FUTURE';

      $('modeBanner').textContent =
        state.baselineFrames.length
          ? (
            state.candidateFrames.length
              ? 'SUPPORTED MODEL COUNTERFACTUAL · NOT OBSERVED'
              : 'FROZEN SAM-WM · RESEARCH FORECAST · NOT OPERATIONALLY CERTIFIED'
          )
          : 'NO MODEL FORECAST LOADED';
    }

    updateTimelineControls();
    requestAnimationFrame(() => {
      A.map.resize();
      B0.map.resize();
      B1.map.resize();
      updateAt(0);
    });
  }

  document.querySelectorAll('[data-mode]').forEach((button) => {
    button.addEventListener('click', async () => {
      const mode = button.dataset.mode;

      if (
        mode === 'predicted'
        && !state.baselineFrames.length
      ) {
        const ok =
          await runResearchPreview();

        if (!ok) return;
      }

      setMode(mode);
    });
  });

  $('replayTab').addEventListener('click', () => {
    log('REPLAY_NOT_AVAILABLE — requires a real implemented intervention with valid pre/post/control evidence.');
    $('predictionStatus').textContent = 'REPLAY_NOT_AVAILABLE';
    $('predictionExplanation').textContent = 'A real replay is never synthesized. Load a validated intervention evaluation first.';
  });

  function updateLegend(domain) {
    if (!domain) {
      $('legendUnavailable').classList.remove('hidden');
      $('legendBody').classList.add('hidden');
      return;
    }
    $('legendUnavailable').classList.add('hidden');
    $('legendBody').classList.remove('hidden');
    const [lo, hi] = domain;
    $('legendHigh').textContent = `${hi.toFixed(1)} °C`;
    $('legendMid').textContent = `${((lo + hi) / 2).toFixed(1)} °C`;
    $('legendLow').textContent = `${lo.toFixed(1)} °C`;
  }

  function clearSvg(svg) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }

  function svgEl(name, attrs = {}) {
    const node = document.createElementNS(svgNS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }

  function colorHex(value, domain) {
    const [lo, hi] = domain;
    const q = clamp((value - lo) / Math.max(1e-9, hi - lo), 0, 1);
    const stops = [
      [35, 87, 137],
      [42, 137, 164],
      [109, 183, 141],
      [246, 223, 110],
      [230, 124, 65],
      [176, 47, 45],
    ];
    const position = q * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(position));
    const alpha = position - index;
    const rgb = [0, 1, 2].map((channel) => Math.round(
      stops[index][channel] * (1 - alpha) + stops[index + 1][channel] * alpha,
    ));
    return `rgb(${rgb.join(',')})`;
  }

  function renderHistogram(values, domain) {
    const svg = $('histogram');
    clearSvg(svg);
    if (!values.length || !domain) return;
    const [lo, hi] = domain;
    const bins = 24;
    const counts = new Array(bins).fill(0);
    for (const value of values) {
      const index = Math.min(bins - 1, Math.floor(((value - lo) / (hi - lo)) * bins));
      counts[Math.max(0, index)] += 1;
    }
    const maxCount = Math.max(...counts, 1);
    const x0 = 8;
    const width = 304;
    const yBase = 103;
    const height = 82;
    counts.forEach((count, index) => {
      const barWidth = width / bins - 1.2;
      const barHeight = (count / maxCount) * height;
      const centerValue = lo + ((index + 0.5) / bins) * (hi - lo);
      svg.appendChild(svgEl('rect', {
        x: x0 + index * (width / bins),
        y: yBase - barHeight,
        width: barWidth,
        height: barHeight,
        rx: 1,
        fill: colorHex(centerValue, domain),
        opacity: 0.9,
      }));
    });
    svg.appendChild(svgEl('line', { x1: 8, x2: 312, y1: 104, y2: 104, stroke: '#3a4d59', 'stroke-width': 1 }));
    const left = svgEl('text', { x: 8, y: 121, fill: '#758d9b', 'font-size': 9 });
    left.textContent = `${lo.toFixed(1)}°`;
    svg.appendChild(left);
    const right = svgEl('text', { x: 312, y: 121, fill: '#758d9b', 'font-size': 9, 'text-anchor': 'end' });
    right.textContent = `${hi.toFixed(1)}°`;
    svg.appendChild(right);
  }

  function meanRow(row) {
    const finite = (row || [])
      .map(Number)
      .filter(Number.isFinite);

    if (!finite.length) return NaN;

    return finite.reduce(
      (sum, value) => sum + value,
      0,
    ) / finite.length;
  }

  function meanSeries(matrix) {
    return (matrix || []).map(meanRow);
  }

  function trajectoryGeometry(series, lo, hi) {
    const values = [
      ...series,
      ...(lo || []),
      ...(hi || []),
    ].filter(Number.isFinite);

    if (!values.length) return null;

    let yLo = Math.min(...values);
    let yHi = Math.max(...values);

    if (Math.abs(yHi - yLo) < 1e-6) {
      yLo -= 0.5;
      yHi += 0.5;
    }

    const padding =
      (yHi - yLo) * 0.10;

    yLo -= padding;
    yHi += padding;

    const point = (value, index, count) => {
      const x =
        18
        + (
          index
          / Math.max(1, count - 1)
        ) * 284;

      const y =
        103
        - (
          (value - yLo)
          / (yHi - yLo)
        ) * 78;

      return [x, y];
    };

    const path = (values_) =>
      values_
        .map((value, index) => {
          const [x, y] =
            point(
              value,
              index,
              values_.length,
            );

          return (
            `${index ? 'L' : 'M'} `
            + `${x.toFixed(2)} `
            + `${y.toFixed(2)}`
          );
        })
        .join(' ');

    return {
      yLo,
      yHi,
      point,
      path,
    };
  }

  function renderPreviewChart(prediction) {
    const svg = $('previewChart');

    clearSvg(svg);

    if (
      !prediction
      ?.baseline_temperature_c
      ?.length
    ) {
      return;
    }

    const baseline =
      meanSeries(
        prediction.baseline_temperature_c,
      );

    const candidate =
      prediction.candidate_temperature_c
        ? meanSeries(
          prediction.candidate_temperature_c,
        )
        : [];

    const geo =
      trajectoryGeometry(
        baseline,
        candidate,
        [],
      );

    if (!geo) return;

    svg.appendChild(
      svgEl(
        'line',
        {
          x1: 18,
          x2: 302,
          y1: 104,
          y2: 104,
          stroke: '#314651',
          'stroke-width': 1,
        },
      ),
    );

    svg.appendChild(
      svgEl(
        'path',
        {
          d: geo.path(baseline),
          fill: 'none',
          stroke: '#67bce8',
          'stroke-width': 2.8,
        },
      ),
    );

    baseline.forEach(
      (value, index) => {
        const [x, y] =
          geo.point(
            value,
            index,
            baseline.length,
          );

        svg.appendChild(
          svgEl(
            'circle',
            {
              cx: x,
              cy: y,
              r: 2.7,
              fill: '#9fdcff',
            },
          ),
        );
      },
    );

    if (candidate.length) {
      svg.appendChild(
        svgEl(
          'path',
          {
            d: geo.path(candidate),
            fill: 'none',
            stroke: '#7ce0a7',
            'stroke-width': 2.4,
          },
        ),
      );
    }

    const title =
      svgEl(
        'text',
        {
          x: 18,
          y: 14,
          fill: '#dce8ee',
          'font-size': 9,
        },
      );

    title.textContent =
      candidate.length
        ? 'SAM-WM baseline + supported action'
        : 'Frozen SAM-WM mean field';

    svg.appendChild(title);

    const left =
      svgEl(
        'text',
        {
          x: 18,
          y: 121,
          fill: '#758d9b',
          'font-size': 9,
        },
      );

    left.textContent = '+1h';

    svg.appendChild(left);

    const right =
      svgEl(
        'text',
        {
          x: 302,
          y: 121,
          fill: '#758d9b',
          'font-size': 9,
          'text-anchor': 'end',
        },
      );

    right.textContent = '+6h';

    svg.appendChild(right);
  }

  function renderUncertaintyChart(prediction) {
    const svg =
      $('uncertaintyChart');

    if (!svg) return;

    clearSvg(svg);

    if (
      !prediction
      ?.baseline_temperature_c
      ?.length
    ) {
      return;
    }

    const mean =
      meanSeries(
        prediction.baseline_temperature_c,
      );

    const low =
      meanSeries(
        prediction.baseline_interval_low_c,
      );

    const high =
      meanSeries(
        prediction.baseline_interval_high_c,
      );

    const geo =
      trajectoryGeometry(
        mean,
        low,
        high,
      );

    if (!geo) return;

    const upperPoints =
      high.map(
        (value, index) =>
          geo.point(
            value,
            index,
            high.length,
          ),
      );

    const lowerPoints =
      low.map(
        (value, index) =>
          geo.point(
            value,
            index,
            low.length,
          ),
      ).reverse();

    const polygon =
      [
        ...upperPoints,
        ...lowerPoints,
      ]
        .map(
          ([x, y]) =>
            `${x.toFixed(2)},${y.toFixed(2)}`,
        )
        .join(' ');

    svg.appendChild(
      svgEl(
        'polygon',
        {
          points: polygon,
          fill: '#7f5ac6',
          opacity: 0.22,
        },
      ),
    );

    svg.appendChild(
      svgEl(
        'path',
        {
          d: geo.path(low),
          fill: 'none',
          stroke: '#9c7ddd',
          'stroke-width': 1.1,
          opacity: 0.75,
        },
      ),
    );

    svg.appendChild(
      svgEl(
        'path',
        {
          d: geo.path(high),
          fill: 'none',
          stroke: '#9c7ddd',
          'stroke-width': 1.1,
          opacity: 0.75,
        },
      ),
    );

    svg.appendChild(
      svgEl(
        'path',
        {
          d: geo.path(mean),
          fill: 'none',
          stroke: '#dfb5ff',
          'stroke-width': 2.5,
        },
      ),
    );

    const title =
      svgEl(
        'text',
        {
          x: 18,
          y: 14,
          fill: '#d9c4f5',
          'font-size': 9,
        },
      );

    title.textContent =
      'mean forecast ± calibrated conformal radius';

    svg.appendChild(title);

    const left =
      svgEl(
        'text',
        {
          x: 18,
          y: 121,
          fill: '#758d9b',
          'font-size': 9,
        },
      );

    left.textContent = '+1h';

    svg.appendChild(left);

    const right =
      svgEl(
        'text',
        {
          x: 302,
          y: 121,
          fill: '#758d9b',
          'font-size': 9,
          'text-anchor': 'end',
        },
      );

    right.textContent = '+6h';

    svg.appendChild(right);
  }

  function resetObservedAnalytics() {
    $('histEmpty').classList.remove('hidden');
    $('histogram').classList.add('hidden');
    ['metricMean', 'metricP95', 'metricMax', 'metricMin', 'metricTiles'].forEach((id) => { $(id).textContent = '—'; });
  }

  function updateObservedAnalytics(fc, interpolated = false) {
    const values = finiteValues(fc, renderTempOf);
    if (!values.length) {
      resetObservedAnalytics();
      return;
    }
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    $('metricMean').textContent = `${mean.toFixed(2)} °C`;
    $('metricP95').textContent = `${percentile(values, 0.95).toFixed(2)} °C`;
    $('metricMax').textContent = `${Math.max(...values).toFixed(2)} °C`;
    $('metricMin').textContent = `${Math.min(...values).toFixed(2)} °C`;
    $('metricTiles').textContent = String(values.length);
    $('histEmpty').classList.add('hidden');
    $('histogram').classList.remove('hidden');
    renderHistogram(values, state.observedDomain || [Math.min(...values), Math.max(...values)]);
    $('histCaption').textContent = interpolated
      ? 'Visual interpolation between real recorded frames; not a new measurement.'
      : 'Real observed FortyGuard tiles only.';
  }

  function updatePredictionAnalytics(prediction) {
    if (!prediction) return;
    $('uncertaintyEmpty').classList.add('hidden');
    $('uncertaintyBody').classList.remove('hidden');
    $('previewEmpty').classList.add('hidden');
    $('previewChart').classList.remove('hidden');
    $('uncDelta').textContent = `${prediction.predicted_delta_c.toFixed(3)} °C`;
    $('uncInterval').textContent = `[${prediction.interval_low_c.toFixed(3)}, ${prediction.interval_high_c.toFixed(3)}] °C`;
    $('uncSupport').textContent = `${(100 * prediction.support_score).toFixed(1)}%`;
    $('supportFill').style.width = `${clamp(prediction.support_score * 100, 0, 100)}%`;
    const actionable = prediction.status === 'PREDICTED' && prediction.support_score >= 0.15;
    const badge = $('actionabilityBadge');
    badge.textContent = actionable ? 'SUPPORTED MODEL OUTPUT' : `${prediction.status} · NOT ACTIONABLE`;
    badge.className = `actionability-badge ${actionable ? 'good' : 'bad'}`;
    renderPreviewChart(prediction);
  }

  function updateMeasuredFieldCard(frame, fc, sourceLabel) {
    const values = finiteValues(fc, observedTempOf);
    $('currentRange').textContent = values.length
      ? `${Math.min(...values).toFixed(2)}–${Math.max(...values).toFixed(2)} °C`
      : 'DATA_UNAVAILABLE';
    $('fieldSource').textContent = sourceLabel;
    $('fieldGeometry').textContent = `${fc?.features?.length || 0} GeoJSON tiles`;
    $('fieldTimestamp').textContent = frame?.timestamp || '—';
    $('fieldActivity').textContent = frame?.activity_id || '—';
    $('fieldHash').textContent = frame?.content_sha256 ? `${frame.content_sha256.slice(0, 10)}…` : '—';
  }

  function aoiFeatureCollection() {
    if (state.aoi.length < 2) return { type:'FeatureCollection', features:[] };
    const coordinates = state.aoi.length >= 3
      ? [[...state.aoi, state.aoi[0]]]
      : [state.aoi];
    const geometry = state.aoi.length >= 3
      ? { type:'Polygon', coordinates }
      : { type:'LineString', coordinates: state.aoi };
    return {
      type:'FeatureCollection',
      features:[{ type:'Feature', properties:{}, geometry }],
    };
  }

  function updateAoiLayer() {
    $('aoiCount').value = String(state.aoi.length);
    const data = aoiFeatureCollection();
    [A.map, B0.map, B1.map].forEach((map) => {
      if (map.loaded() && map.getSource('cw-aoi')) map.getSource('cw-aoi').setData(data);
    });
  }

  $('drawAoi').addEventListener('click', () => {
    if (state.drawing && state.aoi.length >= 3) {
      state.drawing = false;
      $('drawAoi').textContent = 'Redraw AOI';
      log(`AOI closed with ${state.aoi.length} real map vertices.`);
      return;
    }
    state.aoi = [];
    state.drawing = true;
    $('drawAoi').textContent = 'Finish AOI';
    updateAoiLayer();
    log('AOI drawing enabled. Click at least 3 points on the observed real basemap.');
  });

  $('clearAoi').addEventListener('click', () => {
    state.aoi = [];
    state.drawing = false;
    $('drawAoi').textContent = 'Draw AOI';
    updateAoiLayer();
    log('AOI cleared.');
  });

  A.map.on('click', (event) => {
    if (!state.drawing) return;
    state.aoi.push([event.lngLat.lng, event.lngLat.lat]);
    updateAoiLayer();
    log(`AOI point ${state.aoi.length} captured at real map coordinate.`);
  });

  function attachHover(map) {
    map.on('mousemove', 'cw-thermal-fill', (event) => {
      const feature = event.features?.[0];
      if (!feature) return;
      const temperature = Number(feature.properties?.cw_render_temperature_c);
      const id = String(feature.properties?.tile_id ?? feature.id ?? 'unknown');
      const card = $('hoverCard');
      card.innerHTML = `<strong>Tile ${id}</strong><br>${Number.isFinite(temperature) ? `${temperature.toFixed(2)} °C` : 'DATA_UNAVAILABLE'}`;
      const rect = map.getContainer().getBoundingClientRect();
      card.style.left = `${event.point.x + rect.left + 12}px`;
      card.style.top = `${event.point.y + rect.top + 12}px`;
      card.classList.remove('hidden');
      map.getCanvas().style.cursor = 'crosshair';
    });
    map.on('mouseleave', 'cw-thermal-fill', () => {
      $('hoverCard').classList.add('hidden');
      map.getCanvas().style.cursor = '';
    });
  }
  attachHover(A.map);
  attachHover(B0.map);
  attachHover(B1.map);

  function hottestIds(fc, fraction) {
    const rows = (fc?.features || [])
      .map((feature) => [tileId(feature), observedTempOf(feature)])
      .filter(([id, temperature]) => id && Number.isFinite(temperature))
      .sort((a, b) => b[1] - a[1]);
    const count = Math.max(1, Math.ceil(rows.length * fraction));
    return rows.slice(0, count).map(([id]) => id);
  }

  function markPipelineStep(id, ready, detail) {
    const label = $(id);
    if (!label) return;
    const step = label.closest('.pipeline-step');
    step?.classList.toggle('ready', Boolean(ready));
    step?.classList.toggle('blocked', !ready);
    const detailNode = $(`${id}Detail`);
    if (detailNode) detailNode.textContent = detail;
  }

  function refreshEvidenceStage() {
    const ready = state.observedFrames.length > 0;
    markPipelineStep('stageEvidence', ready, ready
      ? `${state.observedFrames.length} real frame(s) loaded in this session`
      : 'No real FortyGuard field loaded in this browser session');
  }

  async function readiness() {
    try {
      const response = await fetch('/api/readiness');
      const data = await response.json();
      const model = data.counterfactual_model || {};

      state.modelReady =
        Boolean(model.ready);

      state.researchPreviewReady =
        Boolean(model.research_preview_ready);

      $('predictedTab').disabled =
        !state.researchPreviewReady
        && !state.baselineFrames.length;

      if (state.modelReady) {
        $('modelStatus').textContent =
          'FORECAST + ACTION READY';
      } else if (state.researchPreviewReady) {
        $('modelStatus').textContent =
          'RESEARCH FORECAST READY';
      } else {
        $('modelStatus').textContent =
          model.status || 'NOT READY';
      }

      const contextReady = Boolean(
        model.context_bundle_ready
        && model.context_manifest_ready
      );

      markPipelineStep(
        'stageDataset',
        contextReady,
        contextReady
          ? `${data.recorded_real_frames || 0} verified real provider frames available`
          : `Need ${model.context_hours_required || 48} consecutive real context hours`,
      );

      markPipelineStep(
        'stageModel',
        state.researchPreviewReady,
        state.researchPreviewReady
          ? `Frozen SAM-WM ${model.model_id || ''} · real-context inference available`
          : (model.status || 'MODEL_NOT_READY'),
      );

      markPipelineStep(
        'stageCalibration',
        Boolean(model.calibration_ready),
        model.calibration_ready
          ? 'Frozen Freiburg validation calibration artifact verified'
          : 'Calibration artifact required',
      );

      markPipelineStep(
        'stageOOD',
        Boolean(model.evaluation_ready),
        model.evaluation_ready
          ? 'Frozen final/OOD evaluation bundle verified'
          : 'Final/OOD evaluation evidence required',
      );

      refreshEvidenceStage();

      if (
        !state.researchPreviewReady
        && state.mode === 'predicted'
      ) {
        setMode('observed');
      }

      if (state.modelReady) {
        $('predictionStatus').textContent =
          'OPERATIONAL MODEL READY';

        $('predictionExplanation').textContent =
          'Frozen SAM-WM and supported action evidence are operationally available.';
      } else if (state.researchPreviewReady) {
        $('predictionStatus').textContent =
          'RESEARCH FORECAST AVAILABLE';

        $('predictionExplanation').textContent =
          'Frozen SAM-WM can forecast from the real FortyGuard context, '
          + 'but the provider replay gate is not certified. '
          + 'Research preview only; cooling actions remain locked.';
      } else {
        $('predictionStatus').textContent =
          model.status || 'MODEL_NOT_READY';

        $('predictionExplanation').textContent =
          'A valid frozen model and real provider context are required.';
      }

      log(
        `readiness: real_frames=${data.recorded_real_frames}; `
        + `research_preview=${state.researchPreviewReady}; `
        + `operational=${state.modelReady}; `
        + `provider=${model.provider_replay_status}`,
      );
    } catch (error) {
      $('modelStatus').textContent =
        'READINESS ERROR';

      log(
        `readiness endpoint failed: ${error.message}`,
      );
    }
  }

  $('loadTimeline').addEventListener('click', async () => {
    try {
      const response = await fetch('/api/evidence/timeline?limit=96');
      const data = await response.json();
      if (!response.ok || !data.frames?.length) {
        log('No compatible recorded real timeline found. Collect repeated real heatmaps first.');
        return;
      }
      state.observedFrames = data.frames;
      refreshEvidenceStage();
      state.geometry = data.frames[data.frames.length - 1].map_data;
      state.gridSignature = data.grid_signature;
      state.observedDomain = temperatureDomain(data.frames, observedTempOf);
      $('sourceStatus').textContent = `REAL TIMELINE · ${data.frames.length}`;
      $('locationLabel').textContent = 'REAL AOI · RECORDED TIMELINE';
      updateLegend(state.observedDomain);
      updateMeasuredFieldCard(data.frames[data.frames.length - 1], state.geometry, 'FortyGuard TCM · recorded evidence');
      fitAll(state.geometry);
      setMode('observed');
      updateAt(0);
      log(`loaded ${data.frames.length} compatible real recorded frames on one verified grid`);
    } catch (error) {
      log(`timeline unavailable: ${error.message}`);
    }
  });

  $('runHeatmap').addEventListener('click', async () => {
    if (state.aoi.length < 3) {
      log('Need at least 3 AOI points on the real map.');
      return;
    }
    const date = $('date').value;
    const time = $('time').value;
    if (!date || !time) {
      log('Date and time are required.');
      return;
    }
    const coordinates = [...state.aoi, state.aoi[0]];
    const payload = {
      polygon_aoi: {
        type: 'FeatureCollection',
        features: [{
          type: 'Feature',
          properties: {},
          geometry: { type: 'Polygon', coordinates: [coordinates] },
        }],
      },
      date_time: { start_date: date, start_time: time, filter_type: 1 },
      granularity: Number($('granularity').value),
      analytic_type: 'tcm',
    };

    $('runHeatmap').disabled = true;
    $('runHeatmap').textContent = 'Waiting for FortyGuard…';
    log('submitting real FortyGuard heatmap request…');
    try {
      const response = await fetch('/api/fortyguard/heatmap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        log(data.detail || 'DATA_UNAVAILABLE');
        return;
      }
      const frame = {
        timestamp: `${date}T${time}:00`,
        activity_id: data.activity_id,
        content_sha256: data.provenance.content_sha256,
        request_sha256: data.provenance.request_sha256,
        map_data: data.map_data,
      };
      state.observedFrames = [frame];
      refreshEvidenceStage();
      state.geometry = data.map_data;
      state.gridSignature = data.grid_signature;
      state.observedDomain = temperatureDomain(state.observedFrames, observedTempOf);
      $('sourceStatus').textContent = 'REAL FORTYGUARD · 1 FRAME';
      $('locationLabel').textContent = 'REAL AOI · FORTYGUARD FIELD';
      updateLegend(state.observedDomain);
      updateMeasuredFieldCard(frame, data.map_data, 'FortyGuard TCM · observed evidence');
      fitAll(data.map_data);
      setMode('observed');
      updateAt(0);
      log(`real activity_id=${data.activity_id}`);
      log(`evidence sha256=${data.provenance.content_sha256}`);
    } catch (error) {
      log(`FortyGuard request failed: ${error.message}`);
    } finally {
      $('runHeatmap').disabled = false;
      $('runHeatmap').textContent = 'Run FortyGuard heatmap';
    }
  });

  $('coverage').addEventListener('input', (event) => {
    $('coverageLabel').textContent = `${event.target.value}%`;
  });

  async function runResearchPreview() {
    if (!state.researchPreviewReady) {
      log(
        'Research preview unavailable: '
        + 'valid frozen model + real 48-hour context required.',
      );
      return false;
    }

    if (
      !state.geometry
      || !state.observedFrames.length
    ) {
      log(
        'Load the recorded real FortyGuard timeline first.',
      );
      return false;
    }

    if (!state.gridSignature) {
      log(
        'Verified real grid signature is missing.',
      );
      return false;
    }

    $('predictionStatus').textContent =
      'RUNNING FROZEN SAM-WM…';

    log(
      'running frozen SAM-WM +1…+6 h '
      + 'research forecast on real FortyGuard context…',
    );

    try {
      const response = await fetch(
        '/api/forecast-preview',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            grid_signature:
              state.gridSignature,
          }),
        },
      );

      const data = await response.json();

      if (!response.ok) {
        $('predictionStatus').textContent =
          data.detail
          || 'RESEARCH_FORECAST_UNAVAILABLE';

        log(
          data.detail
          || 'RESEARCH_FORECAST_UNAVAILABLE',
        );

        return false;
      }

      const prediction = data.prediction;

      state.prediction = prediction;
      state.selectedIds = [];

      state.baselineFrames =
        prediction.future_timestamps.map(
          (timestamp, horizon) => ({
            timestamp,
            map_data: frameFromModelTemps(
              state.geometry,
              prediction.tile_ids,
              prediction.baseline_temperature_c[
                horizon
              ],
            ),
          }),
        );

      // Do not invent intervention/cooling fields.
      state.candidateFrames = [];

      state.predictedDomain =
        temperatureDomain(
          state.baselineFrames,
          modelTempOf,
        );

      updateLegend(
        state.predictedDomain,
      );

      $('uncertaintyEmpty').classList.add(
        'hidden',
      );

      $('uncertaintyBody').classList.remove(
        'hidden',
      );

      $('uncDelta').textContent =
        '+1…+6 h';

      $('uncInterval').textContent =
        `±${Number(
          prediction.baseline_conformal_radius_c,
        ).toFixed(3)} °C`;

      $('uncSupport').textContent =
        '79.90% / 80% gate';

      $('supportFill').style.width =
        '0%';

      const badge =
        $('actionabilityBadge');

      badge.textContent =
        'NOT OPERATIONALLY CERTIFIED';

      badge.className =
        'actionability-badge bad';

      $('previewEmpty').classList.add(
        'hidden',
      );

      $('previewChart').classList.remove(
        'hidden',
      );

      $('uncertaintyChart')?.classList.remove(
        'hidden',
      );

      $('previewCaption').textContent =
        'Actual frozen SAM-WM +1…+6 h baseline forecast. '
        + 'No causal intervention effect is invented.';

      renderPreviewChart(
        prediction,
      );

      renderUncertaintyChart(
        prediction,
      );

      $('predictionStatus').textContent =
        'SAM-WM RESEARCH FORECAST READY';

      $('predictionExplanation').textContent =
        'Actual frozen seed-42 SAM-WM output '
        + 'from the verified real FortyGuard context. '
        + 'Provider replay narrowly missed the fixed '
        + 'operational coverage gate, so this output '
        + 'is research-only and not actionable.';

      $('sourceStatus').textContent =
        'SAM-WM · FROZEN RESEARCH FORECAST';

      $('locationLabel').textContent =
        'SAN JOSÉ · REAL PROVIDER GRID';

      $('predictedTab').disabled =
        false;

      setMode(
        'predicted',
      );

      fitAll(
        state.geometry,
      );

      updateAt(0);

      log(
        `research forecast ready: `
        + `${prediction.future_timestamps.length} horizons; `
        + `${prediction.tile_ids.length} real-grid tiles; `
        + `checkpoint=${prediction.checkpoint_sha256.slice(0, 12)}…`,
      );

      log(
        'truth boundary: MODEL PREDICTION; '
        + 'NOT OBSERVED; NOT OPERATIONALLY CERTIFIED; '
        + 'NOT A CAUSAL COOLING EFFECT.',
      );

      return true;
    } catch (error) {
      $('predictionStatus').textContent =
        'RESEARCH_FORECAST_ERROR';

      log(
        `research forecast unavailable: ${error.message}`,
      );

      return false;
    }
  }


  $('predict').addEventListener('click', async () => {
    if (!state.modelReady) {
      $('predictionStatus').textContent =
        'COOLING ACTION LOCKED';

      $('predictionExplanation').textContent =
        'The frozen SAM-WM baseline research forecast is available, '
        + 'but numerical cooling intervention effects remain blocked '
        + 'without operational replay certification and independent '
        + 'CANDRA action evidence.';

      log(
        'counterfactual cooling blocked — '
        + 'no cooling effect will be fabricated.',
      );

      return;
    }
    if (!state.geometry || !state.observedFrames.length) {
      log('Load a real field or real recorded timeline before prediction.');
      return;
    }
    if (!state.gridSignature) {
      log('Verified grid signature missing; prediction blocked.');
      return;
    }

    const selectedIds = hottestIds(state.geometry, Number($('targetFraction').value));
    if (!selectedIds.length) {
      log('No real temperature-bearing tiles available for action targeting.');
      return;
    }
    state.selectedIds = selectedIds;
    const body = {
      kind: $('actionKind').value,
      grid_signature: state.gridSignature,
      coverage_fraction: Number($('coverage').value) / 100,
      tile_ids: selectedIds,
      cost: null,
    };

    $('predict').disabled = true;
    $('predict').textContent = 'Running model…';
    log(`requesting model future: ${body.kind}; ${selectedIds.length} real tile IDs`);
    try {
      const response = await fetch('/api/counterfactual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) {
        $('predictionStatus').textContent = data.detail || 'COUNTERFACTUAL_UNAVAILABLE';
        log(data.detail || 'COUNTERFACTUAL_UNAVAILABLE');
        return;
      }
      const prediction = data.prediction;
      state.prediction = prediction;
      $('predictedTab').disabled = false;
      state.baselineFrames = prediction.future_timestamps.map((timestamp, horizon) => ({
        timestamp,
        map_data: frameFromModelTemps(state.geometry, prediction.tile_ids, prediction.baseline_temperature_c[horizon]),
      }));
      state.candidateFrames = prediction.future_timestamps.map((timestamp, horizon) => ({
        timestamp,
        map_data: frameFromModelTemps(state.geometry, prediction.tile_ids, prediction.candidate_temperature_c[horizon]),
      }));
      state.predictedDomain = mergedPredictionDomain();
      updatePredictionAnalytics(prediction);
      $('predictionStatus').textContent = prediction.status;
      $('predictionExplanation').textContent = prediction.status === 'PREDICTED'
        ? 'Model output is support-gated and remains explicitly labelled as prediction.'
        : `${prediction.status}. The result may be inspected as model output but is NOT ACTIONABLE.`;
      setMode('predicted');
      fitAll(state.geometry);
      updateAt(0);
      log(`model status=${prediction.status}; mean_delta=${prediction.predicted_delta_c.toFixed(4)} °C`);
    } catch (error) {
      log(`counterfactual unavailable: ${error.message}`);
    } finally {
      $('predict').disabled = false;
      $('predict').textContent = 'Predict cooling future';
    }
  });

  ['toggleBuildings', 'toggleThermal', 'toggleAction'].forEach((id) => {
    $(id).addEventListener('change', () => {
      [A.map, B0.map, B1.map].forEach(applyLayerVisibility);
    });
  });

  function resetCamera() {
    [A.map, B0.map, B1.map].forEach((map) => map.easeTo({ ...DEFAULT_VIEW, duration: 900 }));
  }
  $('resetCamera').addEventListener('click', resetCamera);

  function orbitStep(timestamp) {
    if (!state.orbiting) return;
    if (!orbitStep.last) orbitStep.last = timestamp;
    const dt = timestamp - orbitStep.last;
    orbitStep.last = timestamp;
    const targets =
      state.mode === 'observed'
      || !state.candidateFrames.length
        ? [A.map]
        : [B0.map, B1.map];
    targets.forEach((map) => map.rotateTo(map.getBearing() + dt * 0.006, { duration: 0 }));
    state.orbitRaf = requestAnimationFrame(orbitStep);
  }
  orbitStep.last = 0;

  $('orbitButton').addEventListener('click', () => {
    state.orbiting = !state.orbiting;
    $('orbitButton').textContent = state.orbiting ? 'Stop 3D camera orbit' : 'Start 3D camera orbit';
    $('orbitTruth').textContent = state.orbiting
      ? 'CAMERA ORBIT ACTIVE · visualization only; temperature data are unchanged.'
      : 'Camera motion is visualization only.';
    orbitStep.last = 0;
    if (state.orbiting) {
      const targets =
      state.mode === 'observed'
      || !state.candidateFrames.length
        ? [A.map]
        : [B0.map, B1.map];
      if (!state.geometry) targets.forEach((map) => map.easeTo({ ...DEFAULT_VIEW, duration: 500 }));
      state.orbitRaf = requestAnimationFrame(orbitStep);
    } else if (state.orbitRaf) {
      cancelAnimationFrame(state.orbitRaf);
    }
  });

  function setDefaultDateTime() {
    const now = new Date();
    const localDate = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const localTime = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    $('date').value = localDate;
    $('time').value = localTime;
  }

  resetObservedAnalytics();
  updateLegend(null);
  setDefaultDateTime();
  updateTimelineControls();
  refreshEvidenceStage();
  readiness();
  setMode('observed');

  // Load the already-recorded real FortyGuard timeline
  // automatically. This makes ZERO provider API calls.
  window.setTimeout(() => {
    if (
      !state.observedFrames.length
      && $('loadTimeline')
    ) {
      $('loadTimeline').click();
    }
  }, 450);
  log(
    'renderer booted: real observed field first; '
    + 'frozen SAM-WM research preview is shown only '
    + 'when real context is available.',
  );
})();
