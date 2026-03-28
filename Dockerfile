# ─────────────────────────────────────────
# Dockerfile
# HF Spaces-compatible multi-stage build
# Port 7860 is required by HF Spaces
# ─────────────────────────────────────────

FROM python:3.11-slim

# HF Spaces requires non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# install system dependencies
RUN pip install --no-cache-dir --upgrade pip

# copy requirements first (layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files
COPY --chown=user . /app

# HF Spaces requires port 7860
EXPOSE 7860

# start server
CMD ["python", "server.py"]
