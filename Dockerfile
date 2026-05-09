# Multi-stage build for local-livekit-plugins voice agent
# Optimized for minimal image size and fast startup

FROM python:3.12-slim as builder

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY examples/ examples/
COPY models/ models/

COPY README.md ./
# Install dependencies with uv
RUN uv sync --frozen --no-dev

# Final runtime stage
FROM python:3.12-slim

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY examples/ examples/
COPY models/ models/
COPY examples/.env.local examples/.env.local

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command - run optimized voice agent in console mode
CMD ["python", "-m", "examples.voice_agent_optimized", "console"]
