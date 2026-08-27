FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY static ./static
COPY train.py eval.py fortyguard_check.py plot.py summarize.py ./
RUN python -m pip install --no-cache-dir '.[app]'
RUN useradd --create-home --uid 10001 samwm && chown -R samwm:samwm /app
USER samwm
EXPOSE 8000
CMD ["uvicorn", "coolworld.app:app", "--host", "0.0.0.0", "--port", "8000"]
