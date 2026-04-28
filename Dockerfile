# Use Python 3.11 slim — confirmed to have TensorFlow wheels on Linux
FROM python:3.11-slim

# Install system dependencies (ffmpeg for pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app files
COPY app.py .
COPY index.html .

# Expose port
EXPOSE 5000

# Start with gunicorn — 1 worker to stay within 512 MB free tier RAM
CMD gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
