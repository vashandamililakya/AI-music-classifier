#!/usr/bin/env bash
set -e

# Print Python version to help debug any future pip issues
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip --version)"

# Install ffmpeg for pydub (MP3 / M4A / AAC / WebM decoding)
apt-get update -y && apt-get install -y ffmpeg

# Upgrade pip first to avoid resolution issues
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
