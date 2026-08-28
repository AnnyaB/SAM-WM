FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    WEB_CONCURRENCY=1 \
    COOLWORLD_LIVE_API_ENABLED=0

WORKDIR /app

RUN useradd --create-home --uid 10001 samwm

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY static ./static
COPY artifacts/deployment ./artifacts/deployment
COPY artifacts/fortyguard ./artifacts/fortyguard
COPY artifacts/summary.json ./artifacts/summary.json
COPY artifacts/DEPLOYMENT_SELECTION.json ./artifacts/DEPLOYMENT_SELECTION.json
COPY artifacts/FREEZE_MANIFEST.json ./artifacts/FREEZE_MANIFEST.json
COPY artifacts/FAIRURBTEMP_PREREG.json ./artifacts/FAIRURBTEMP_PREREG.json
COPY artifacts/FROZEN_SOURCE_SHA.txt ./artifacts/FROZEN_SOURCE_SHA.txt

RUN python -m pip install --no-cache-dir '.[app]' \
    && chown -R samwm:samwm /app

USER samwm

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '7860') + '/api/health', timeout=4).read()" || exit 1

CMD ["sh", "-c", "uvicorn coolworld.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers ${WEB_CONCURRENCY:-1} --proxy-headers --forwarded-allow-ips='*'"]
