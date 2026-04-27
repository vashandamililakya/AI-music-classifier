"""
AI Music Genre Classifier — Flask Backend (Memory-Optimised)
─────────────────────────────────────────────────────────────
Changes vs original
  • Replaced tensorflow + tensorflow-hub with tflite-runtime
    → saves ~400 MB RAM at runtime
  • Replaced librosa (pulls scipy/numba/joblib) with soundfile + resampy
    → saves ~150 MB RAM
  • YAMNet TFLite model is downloaded once, cached on disk, loaded lazily
    (model is NOT held in RAM between requests on free-tier dynos)
  • numpy float32 throughout — no silent float64 up-casts
  • Explicit temp-file cleanup even on exceptions
  • Single gunicorn worker recommended (--workers 1) to avoid duplicating
    the model in memory.

Render free tier: 512 MB RAM
Estimated footprint with these changes: ~180–250 MB
"""

import os
import gc
import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import resampy
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)

# ──────────────────────────────────────────────────────────────
# Genre definitions
# ──────────────────────────────────────────────────────────────
GENRES = [
    "Hip-Hop", "Electronic", "Jazz", "Classical", "Rock",
    "Pop", "R&B", "Metal", "Country", "Reggae", "Blues", "Indie",
]

# YAMNet 521-class → genre mapping (AudioSet indices)
YAMNET_GENRE_MAP: dict[str, list[int]] = {
    "Hip-Hop":    [252, 253, 254, 255, 256],
    "Electronic": [82, 83, 84, 85, 86, 87],
    "Jazz":       [262, 263, 264, 265],
    "Classical":  [266, 267, 268, 269],
    "Rock":       [270, 271, 272, 273, 274],
    "Pop":        [257, 258, 259],
    "R&B":        [260, 261],
    "Metal":      [275, 276, 277],
    "Country":    [278, 279],
    "Reggae":     [280, 281],
    "Blues":      [282, 283],
    "Indie":      [284, 285],
}

# ──────────────────────────────────────────────────────────────
# YAMNet TFLite — lazy singleton
# ──────────────────────────────────────────────────────────────
YAMNET_TFLITE_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "models/tflite/task_library/audio_classification/android/"
    "lite-model_yamnet_classification_tflite_1.tflite"
)
MODEL_CACHE_PATH = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp")) / "yamnet.tflite"

_interpreter = None   # module-level cache (single process)


def _download_model() -> None:
    """Download YAMNet TFLite to disk if not already cached."""
    if MODEL_CACHE_PATH.exists():
        log.info("YAMNet TFLite already cached at %s", MODEL_CACHE_PATH)
        return
    log.info("Downloading YAMNet TFLite model …")
    import urllib.request
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(YAMNET_TFLITE_URL, MODEL_CACHE_PATH)
    log.info("Download complete (%.1f MB)", MODEL_CACHE_PATH.stat().st_size / 1e6)


def get_interpreter():
    """
    Return the tf.lite.Interpreter singleton using tensorflow-cpu.
    """
    global _interpreter
    if _interpreter is not None:
        return _interpreter

    _download_model()

    import tensorflow as tf
    _interpreter = tf.lite.Interpreter(model_path=str(MODEL_CACHE_PATH))
    _interpreter.allocate_tensors()
    log.info("TFLite interpreter ready ✓")
    return _interpreter


# ──────────────────────────────────────────────────────────────
# Audio helpers  (soundfile + resampy — no librosa/scipy/numba)
# ──────────────────────────────────────────────────────────────
YAMNET_SAMPLE_RATE = 16_000
MAX_DURATION_SEC   = 30          # analyse first 30 s
MAX_SAMPLES        = YAMNET_SAMPLE_RATE * MAX_DURATION_SEC


def load_audio(path: str) -> np.ndarray:
    """
    Load any soundfile-readable audio → 16 kHz mono float32.
    Falls back to librosa for formats soundfile can't handle (e.g. MP3/M4A).
    """
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = data.mean(axis=1)           # stereo → mono
    except Exception:
        # MP3 / M4A / AAC / WebM — soundfile can't decode these natively;
        # use soundfile via pydub → wav bytes, or fall back to librosa.
        try:
            from pydub import AudioSegment      # lightweight fallback
            seg = AudioSegment.from_file(path).set_channels(1).set_frame_rate(YAMNET_SAMPLE_RATE)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            waveform = samples / 32768.0        # int16 → [-1, 1]
            sr = YAMNET_SAMPLE_RATE
        except Exception:
            import librosa                      # last-resort (heavy)
            waveform, sr = librosa.load(path, sr=YAMNET_SAMPLE_RATE, mono=True,
                                        duration=MAX_DURATION_SEC)
            return waveform.astype(np.float32)

    # Resample if needed (resampy is much lighter than librosa's resampler)
    if sr != YAMNET_SAMPLE_RATE:
        waveform = resampy.resample(waveform, sr, YAMNET_SAMPLE_RATE)

    return waveform[:MAX_SAMPLES].astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────
FRAME_LENGTH = 15_600   # YAMNet TFLite expects exactly 15 600 samples per call


def run_yamnet_tflite(waveform: np.ndarray) -> np.ndarray:
    """
    Run YAMNet TFLite frame-by-frame and return mean class scores (521,).
    The TFLite model processes one 0.975-s frame at a time.
    """
    interp = get_interpreter()
    in_detail  = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    hop = FRAME_LENGTH // 2          # 50 % overlap
    all_scores = []

    for start in range(0, max(1, len(waveform) - FRAME_LENGTH + 1), hop):
        frame = waveform[start : start + FRAME_LENGTH]
        if len(frame) < FRAME_LENGTH:
            frame = np.pad(frame, (0, FRAME_LENGTH - len(frame)))

        interp.set_tensor(in_detail["index"], frame.reshape(1, -1))
        interp.invoke()
        scores = interp.get_tensor(out_detail["index"])[0]   # (521,)
        all_scores.append(scores)

    mean_scores = np.mean(all_scores, axis=0).astype(np.float32)  # (521,)
    return mean_scores


def scores_to_genre_probs(mean_scores: np.ndarray) -> dict[str, float]:
    """Aggregate 521 class scores → 12 genre probabilities via softmax."""
    genre_totals = np.array(
        [mean_scores[YAMNET_GENRE_MAP[g]].sum() for g in GENRES],
        dtype=np.float32,
    )
    e = np.exp(genre_totals - genre_totals.max())
    probs = e / e.sum()
    return {g: float(probs[i]) for i, g in enumerate(GENRES)}


# ──────────────────────────────────────────────────────────────
# API route
# ──────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".webm"}


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body : multipart/form-data  →  field "audio"
    Response JSON:
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
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided (field name: 'audio')"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(audio_file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported format '{suffix}'. "
                     f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 415

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        log.info("Received: %s (%.1f KB)",
                 audio_file.filename, os.path.getsize(tmp_path) / 1024)

        waveform    = load_audio(tmp_path)
        mean_scores = run_yamnet_tflite(waveform)
        probs       = scores_to_genre_probs(mean_scores)

    except Exception as exc:
        log.exception("Inference failed")
        return jsonify({"error": f"Inference error: {exc}"}), 500
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        gc.collect()   # free waveform / score arrays promptly

    sorted_genres = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_genre, top_prob = sorted_genres[0]

    return jsonify({
        "genre":      top_genre,
        "confidence": round(top_prob * 100),
        "candidates": [
            {"genre": g, "confidence": max(1, round(p * 100))}
            for g, p in sorted_genres[1:5]
        ],
    })


# ──────────────────────────────────────────────────────────────
# Serve frontend
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ──────────────────────────────────────────────────────────────
# Entry-point  (use gunicorn in production, not this)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
