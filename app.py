"""
AI Music Genre Classifier — Flask Backend
Architecture : YAMNet (transfer learning via TensorFlow Hub)
Endpoint     : POST /api/predict  — accepts multipart audio file
               GET  /             — serves index.html
"""

import os
import io
import tempfile
import logging
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)  # allow requests from the frontend dev server

# ──────────────────────────────────────────────
# Genre definitions
# ──────────────────────────────────────────────
GENRES = [
    "Hip-Hop", "Electronic", "Jazz", "Classical", "Rock",
    "Pop", "R&B", "Metal", "Country", "Reggae", "Blues", "Indie",
]

# ──────────────────────────────────────────────
# YAMNet → Genre mapping
#
# YAMNet outputs 521 AudioSet classes.  We map the most
# musically-relevant AudioSet class indices to our 12 genres.
# (https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv)
# ──────────────────────────────────────────────
YAMNET_GENRE_MAP: dict[str, list[int]] = {
    "Hip-Hop":    [252, 253, 254, 255, 256],   # Hip hop, Rap, Beatboxing …
    "Electronic": [82, 83, 84, 85, 86, 87],    # Electronic music, Techno, House …
    "Jazz":       [262, 263, 264, 265],         # Jazz, Bebop, Swing …
    "Classical":  [266, 267, 268, 269],         # Classical, Opera, Orchestra …
    "Rock":       [270, 271, 272, 273, 274],    # Rock, Hard rock, Punk …
    "Pop":        [257, 258, 259],              # Pop music …
    "R&B":        [260, 261],                   # Rhythm and blues, Soul …
    "Metal":      [275, 276, 277],              # Heavy metal, Death metal …
    "Country":    [278, 279],                   # Country, Bluegrass …
    "Reggae":     [280, 281],                   # Reggae, Ska …
    "Blues":      [282, 283],                   # Blues …
    "Indie":      [284, 285],                   # Indie, Alternative …
}

# Flat lookup: yamnet_class_index → genre_name
_CLASS_TO_GENRE: dict[int, str] = {}
for _genre, _indices in YAMNET_GENRE_MAP.items():
    for _idx in _indices:
        _CLASS_TO_GENRE[_idx] = _genre

# ──────────────────────────────────────────────
# Load YAMNet once at startup (cached on disk by TF Hub)
# ──────────────────────────────────────────────
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

log.info("Loading YAMNet from TF Hub …")
yamnet_model = hub.load(YAMNET_MODEL_URL)
log.info("YAMNet loaded ✓")

# ──────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────
YAMNET_SAMPLE_RATE = 16_000   # YAMNet expects 16 kHz mono PCM
MAX_DURATION_SEC   = 30       # analyse only the first 30 s for speed


def load_audio(path: str) -> np.ndarray:
    """Load any audio file supported by librosa → 16 kHz mono float32."""
    waveform, sr = librosa.load(
        path,
        sr=YAMNET_SAMPLE_RATE,
        mono=True,
        duration=MAX_DURATION_SEC,
    )
    return waveform.astype(np.float32)


def run_yamnet(waveform: np.ndarray) -> np.ndarray:
    """
    Run YAMNet inference.

    Returns
    -------
    mean_scores : ndarray, shape (521,)
        Mean per-class scores averaged over all 0.48-s frames.
    """
    scores, _embeddings, _spectrogram = yamnet_model(waveform)
    # scores shape: (num_frames, 521)
    mean_scores: np.ndarray = tf.reduce_mean(scores, axis=0).numpy()
    return mean_scores


def scores_to_genre_probs(mean_scores: np.ndarray) -> dict[str, float]:
    """
    Aggregate the 521 YAMNet class scores into 12 genre probabilities.

    Strategy
    --------
    1. For each genre, sum the scores of its mapped AudioSet classes.
    2. Softmax the 12 genre totals to get a proper probability distribution.
    """
    genre_totals = np.zeros(len(GENRES), dtype=np.float32)
    for i, genre in enumerate(GENRES):
        indices = YAMNET_GENRE_MAP[genre]
        genre_totals[i] = mean_scores[indices].sum()

    # Softmax for nicer probabilities
    e = np.exp(genre_totals - genre_totals.max())
    probs = e / e.sum()
    return {genre: float(probs[i]) for i, genre in enumerate(GENRES)}


# ──────────────────────────────────────────────
# API route
# ──────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".webm"}


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body : multipart/form-data  →  field "audio"
    Response (JSON):
      {
        "genre":      "Rock",
        "confidence": 87,
        "candidates": [
          {"genre": "Metal",    "confidence": 6},
          {"genre": "Indie",    "confidence": 4},
          {"genre": "Blues",    "confidence": 2},
          {"genre": "Pop",      "confidence": 1}
        ]
      }
    """
    # ── Validate upload ──────────────────────────────────
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided (field name: 'audio')"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(audio_file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported format '{suffix}'. "
                     f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 415

    # ── Save to temp file so librosa can read it ─────────
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        log.info("Received file: %s (%.1f KB)",
                 audio_file.filename, os.path.getsize(tmp_path) / 1024)

        # ── Inference ────────────────────────────────────
        waveform    = load_audio(tmp_path)
        mean_scores = run_yamnet(waveform)
        probs       = scores_to_genre_probs(mean_scores)

    except Exception as exc:
        log.exception("Inference failed")
        return jsonify({"error": f"Inference error: {exc}"}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # ── Build response ───────────────────────────────────
    sorted_genres = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_genre, top_prob = sorted_genres[0]
    top_conf = round(top_prob * 100)

    candidates = [
        {"genre": g, "confidence": max(1, round(p * 100))}
        for g, p in sorted_genres[1:5]   # next 4 genres
    ]

    response_payload = {
        "genre":      top_genre,
        "confidence": top_conf,
        "candidates": candidates,
    }
    log.info("Prediction → %s (%d%%)", top_genre, top_conf)
    return jsonify(response_payload)


# ──────────────────────────────────────────────
# Serve the frontend (index.html)
# ──────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the HTML frontend from the same directory."""
    return send_from_directory(app.static_folder, "index.html")


# ──────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
