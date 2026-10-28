# syntax=docker/dockerfile:1
# vessel-eta API: docker build -t vessel-eta . && docker run -p 8000:8000 vessel-eta

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# dependencies first: this layer is reused while only the code changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim
# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY configs ./configs
COPY artifacts ./artifacts
ENV PATH="/app/.venv/bin:$PATH" \
    VESSEL_ETA_ROOT=/app \
    VESSEL_ETA_MODEL=/app/artifacts/eta_v2 \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "vessel_eta.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
