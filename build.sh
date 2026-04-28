#!/usr/bin/env bash
set -e

echo "Python version: $(python3 --version)"
echo "Pip version: $(pip --version)"

# Install ffmpeg for pydub
apt-get update -y && apt-get install -y ffmpeg

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
