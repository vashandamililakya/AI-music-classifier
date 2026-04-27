#!/usr/bin/env bash
set -e

# Install ffmpeg for pydub (MP3 / M4A / AAC / WebM decoding)
apt-get update -y && apt-get install -y ffmpeg

# Install Python dependencies
pip install -r requirements.txt
