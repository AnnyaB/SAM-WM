FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md DATA_POLICY.md SOURCES.md AGENTS.md ./
COPY src ./src
COPY configs ./configs
COPY static ./static

RUN python -m pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 coolworld && chown -R coolworld:coolworld /app
USER coolworld

EXPOSE 8000
CMD ["uvicorn", "coolworld.api:app", "--host", "0.0.0.0", "--port", "8000"]
