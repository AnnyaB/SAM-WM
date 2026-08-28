(() => {
  'use strict';

  const overlay = document.getElementById('dependencyOverlay');
  const title = document.getElementById('dependencyTitle');
  const detail = document.getElementById('dependencyDetail');

  const setStatus = (heading, message, isError = false) => {
    title.textContent = heading;
    detail.textContent = message;
    overlay.classList.toggle('error', isError);
  };

  const loadStylesheet = (urls) => new Promise((resolve) => {
    let index = 0;
    const attempt = () => {
      if (index >= urls.length) {
        resolve(false);
        return;
      }
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = urls[index++];
      link.onload = () => resolve(true);
      link.onerror = () => {
        link.remove();
        attempt();
      };
      document.head.appendChild(link);
    };
    attempt();
  });

  const loadScript = (urls, globalName) => new Promise((resolve, reject) => {
    let index = 0;
    const attempt = () => {
      if (window[globalName]) {
        resolve();
        return;
      }
      if (index >= urls.length) {
        reject(new Error(`${globalName} could not be loaded from either CDN`));
        return;
      }
      const script = document.createElement('script');
      script.src = urls[index++];
      script.async = true;
      script.onload = () => {
        if (window[globalName]) {
          resolve();
        } else {
          script.remove();
          attempt();
        }
      };
      script.onerror = () => {
        script.remove();
        attempt();
      };
      document.head.appendChild(script);
    };
    attempt();
  });

  const loadLocalScript = (src, label) => new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`${label} could not be loaded`));
    document.body.appendChild(script);
  });

  const loadApp = () => loadLocalScript('/static/app.js?v=0.8.0', 'Local app.js');
  const loadInterpretability = () => loadLocalScript(
    '/static/interpretability.js?v=1.1.0',
    'CoolWorld product experience',
  );

  (async () => {
    try {
      setStatus('Loading CoolWorld…', 'Preparing the 3D city map and SAM-WM interface.');
      await Promise.all([
        loadStylesheet([
          'https://unpkg.com/maplibre-gl@5.23.0/dist/maplibre-gl.css',
          'https://cdn.jsdelivr.net/npm/maplibre-gl@5.23.0/dist/maplibre-gl.css',
        ]),
        loadScript([
          'https://unpkg.com/maplibre-gl@5.23.0/dist/maplibre-gl.js',
          'https://cdn.jsdelivr.net/npm/maplibre-gl@5.23.0/dist/maplibre-gl.js',
        ], 'maplibregl'),
      ]);
      setStatus('City view ready', 'Starting the SAM-WM forecast experience.');
      await loadApp();
      await loadInterpretability();
      window.setTimeout(() => overlay.classList.add('hidden'), 250);
    } catch (error) {
      setStatus(
        'MAP RENDERER UNAVAILABLE',
        `${error.message}. Check your internet connection or content blocker and reload.`,
        true,
      );
    }
  })();
})();
