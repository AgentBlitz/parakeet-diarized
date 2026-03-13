# === Stage 1: Builder ===
# Use devel image for compiling C extensions in NeMo/pyannote dependencies
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create venv and install dependencies (cached unless requirements.txt changes)
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install torch with CUDA 12.1 wheels first to ensure GPU support
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt


# === Stage 2: Runtime ===
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy pre-built venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .

# Ensure entrypoint is executable and temp dir exists
RUN chmod +x /app/docker-entrypoint.sh && \
    mkdir -p /tmp/parakeet

EXPOSE 8000 8001

ENTRYPOINT ["/app/docker-entrypoint.sh"]
