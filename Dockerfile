FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY train.py eval.py fortyguard_check.py plot.py ./
RUN pip install --no-cache-dir .
CMD ["python", "-c", "import coolworld; print(coolworld.__version__)"]
