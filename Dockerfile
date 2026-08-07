# ═══════════════════════════════════════════════════════════════
# Aerovigil AI: Physics-Guided BNN for Wind Turbine RUL Prediction
# Multi-stage Dockerfile for CPU and GPU inference
# ═══════════════════════════════════════════════════════════════

# ─── BUILD ARGUMENTS ──────────────────────────────────────────
ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.1
ARG TORCH_VERSION=2.2.0

# ═══════════════════════════════════════════════════════════════
# STAGE 1: Builder (heavy dependencies compilation)
# ═══════════════════════════════════════════════════════════════
FROM python:${PYTHON_VERSION}-slim as builder

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Install Python dependencies first (cache layer)
COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install build \
    && python -m build --wheel --outdir dist/

# ═══════════════════════════════════════════════════════════════
# STAGE 2: Runtime (minimal production image)
# ═══════════════════════════════════════════════════════════════
FROM python:${PYTHON_VERSION}-slim as runtime

LABEL maintainer="Aerovigil AI <contact@aerovigil.ai>"
LABEL org.opencontainers.image.title="Aerovigil PG-BNN"
LABEL org.opencontainers.image.description="Physics-Guided Bayesian Neural Network for Wind Turbine RUL Prediction"
LABEL org.opencontainers.image.source="https://github.com/rajaram-2005/wind-turbine-pg-bnn"
LABEL org.opencontainers.image.licenses="MIT"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV APP_HOME=/app
ENV MODEL_PATH=/app/artifacts/pg_bnn_demo/bnn_demo.pt
ENV CONFIG_PATH=/app/artifacts/pg_bnn_demo/config.json
ENV SCALER_PATH=/app/artifacts/pg_bnn_demo/scaler.npz
ENV PORT=8000

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r aerovigil && useradd -r -g aerovigil -d ${APP_HOME} -s /sbin/nologin aerovigil

# Set working directory
WORKDIR ${APP_HOME}

# Copy and install the built wheel from builder stage
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl

# Copy application code
COPY src/ ${APP_HOME}/src/
COPY config.json /app/config.json
COPY artifacts/ ${APP_HOME}/artifacts/

# Copy and set permissions for entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set ownership
RUN chown -R aerovigil:aerovigil ${APP_HOME}

# Switch to non-root user
USER aerovigil

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Expose port
EXPOSE ${PORT}

# Entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# ═══════════════════════════════════════════════════════════════
# STAGE 3: GPU Runtime (CUDA-enabled for GPU inference)
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 as gpu

ARG PYTHON_VERSION=3.11
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV MODEL_PATH=/app/artifacts/pg_bnn_demo/bnn_demo.pt
ENV CONFIG_PATH=/app/artifacts/pg_bnn_demo/config.json
ENV SCALER_PATH=/app/artifacts/pg_bnn_demo/scaler.npz
ENV PORT=8000

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aerovigil && useradd -r -g aerovigil -d ${APP_HOME} -s /sbin/nologin aerovigil

WORKDIR ${APP_HOME}

# Install PyTorch with CUDA
RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu121

# Copy application
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

COPY src/ ${APP_HOME}/src/
COPY config.json /app/config.json
COPY artifacts/ ${APP_HOME}/artifacts/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN chown -R aerovigil:aerovigil ${APP_HOME}
USER aerovigil

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

EXPOSE ${PORT}
ENTRYPOINT ["/entrypoint.sh"]
