# ============================================================
# Dockerfile — EDU Assist (Production Setup)
# ============================================================

# Stage 1: Build & Dependency Installation
FROM python:3.10-slim AS builder

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for compilation, curl (healthcheck), and FAISS (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & install CPU-only PyTorch first
# This saves ~2.3GB of Docker image size compared to the default GPU-enabled PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Final Light-weight Image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache/huggingface

WORKDIR /app

# Install runtime dependencies: OpenMP for FAISS and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app folders for mounts and caches
RUN mkdir -p /app/data /app/cache/huggingface /app/config

# Create a secure, non-root user and group (UID 1000 is standard)
RUN groupadd -g 1000 appgroup && \
    useradd -r -u 1000 -g appgroup -d /app -s /sbin/nologin appuser

# Copy codebase
COPY admin/ ./admin/
COPY agents/ ./agents/
COPY config/ ./config/
COPY static/ ./static/
COPY utils/ ./utils/
COPY main.py pipeline.py ./

# Secure file permissions for appuser
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Expose FastAPI default port
EXPOSE 8000

# Health check to ensure API is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Start the application
# We use 1 worker in production because loading LLMs and Embeddings in memory takes ~2GB RAM.
# Multiple workers would duplicate memory consumption and can overload standard host VMs.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
