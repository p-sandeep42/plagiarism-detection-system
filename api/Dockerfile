# Use slim python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKERS=1

# Set work directory
WORKDIR /app

# Install system dependencies (libpq-dev removed — no PostgreSQL used)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port
EXPOSE 8000

# Health check so Render knows when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run gunicorn — workers and timeout are configurable via env vars
CMD sh -c "gunicorn main:app -w ${WORKERS} -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120"
