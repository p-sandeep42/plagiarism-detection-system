# Use slim python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Removed ML pre-download to save space and rely on TF-IDF on Render Free Tier)

# Copy project
COPY . .

# Expose port
EXPOSE 8000

# Run gunicorn (Optimized for Render Free Tier: 1 worker, 120s timeout for large batches)
CMD ["sh", "-c", "gunicorn main:app -w ${WORKERS:-1} -k uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:8000"]
