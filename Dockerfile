# Use official Python 3.11 slim image (Debian-based, small & secure)
FROM python:3.11-slim

# Only install build deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install CPU-only PyTorch + other deps
# Note: We install torch separately to use the correct index
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy entire application
COPY . .

# Copy entrypoint with explicit permissions (fixes Windows + simplifies logic)
COPY --chmod=755 entrypoint.sh .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports (for documentation; not strictly needed)
EXPOSE 5001 5002
ENTRYPOINT ["./entrypoint.sh"]