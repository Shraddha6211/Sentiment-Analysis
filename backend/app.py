"""
=============================================================================
app.py — Flask REST API Server
=============================================================================
"""

import os
import sys
import logging
from datetime import datetime

from flask      import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

from model.predict import SentimentPredictor   # ✅ FIXED: correct class name

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentiment-api")

# =============================================================================
# Flask App Initialization
# =============================================================================
app = Flask(__name__)
CORS(app)

# =============================================================================
# Model Loading (at startup, not per request)
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "model", "sentiment_model.pkl")

# ✅ FIXED: Use SentimentPredictor with correct path
#           model_loaded flag tracks whether it loaded successfully
logger.info("Loading sentiment model...")
model_loaded = False
try:
    predictor    = SentimentPredictor(MODEL_PATH)
    model_loaded = True
    logger.info("✅ Model loaded successfully.")
except FileNotFoundError:
    predictor    = None
    model_loaded = False
    logger.warning("⚠️  No trained model found. Run 'python model/train.py' first.")

# =============================================================================
# API Metadata
# =============================================================================
API_VERSION = "1.0.0"
START_TIME  = datetime.utcnow().isoformat() + "Z"

EXAMPLE_TEXTS = [
    "I absolutely love this new phone! The camera is incredible and battery life is amazing!",
    "The service was terrible and the food was cold. I will never come back to this restaurant.",
    "Just got my order delivered. It arrived on time. Nothing special to report.",
    "Can't believe how bad this movie was. Total waste of money. Don't watch it!",
    "The weather today is pretty average, not too hot, not too cold.",
    "This is the best purchase I've made all year! Highly recommend to everyone!",
    "Stuck in traffic again. This city's infrastructure really needs improvement.",
    "The new update fixed some bugs but introduced new ones. Overall, it's okay I guess.",
]

# =============================================================================
# Helper: Consistent response envelopes
# =============================================================================

def make_success_response(data: dict, status_code: int = 200):
    return jsonify({
        "success":   True,
        "data":      data,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), status_code


def make_error_response(message: str, status_code: int = 400):
    return jsonify({
        "success":   False,
        "error":     message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), status_code


# =============================================================================
# Endpoints
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    return make_success_response({
        "service":     "Sentiment Analysis API",
        "version":     API_VERSION,
        "status":      "running",
        "model_ready": model_loaded,
    })


@app.route("/health", methods=["GET"])
def health():
    return make_success_response({
        "status":      "healthy",
        "api_version": API_VERSION,
        "model_ready": model_loaded,
        "started_at":  START_TIME,
        "labels":      ["negative", "neutral", "positive"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return make_error_response("Request must be JSON.", status_code=415)

    body = request.get_json()
    text = body.get("text", "").strip()

    if not text:
        return make_error_response("'text' field is required.", status_code=400)
    if len(text) > 5000:
        return make_error_response("Text exceeds 5000 character limit.", status_code=413)
    if not model_loaded:
        return make_error_response("Model not ready. Run model/train.py first.", status_code=503)

    try:
        result = predictor.predict(text)   # ✅ FIXED: call predictor.predict()
        logger.info(f"Predict: '{text[:40]}' → {result['sentiment']} ({result['confidence']:.2%})")
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return make_error_response(str(e), status_code=500)

    return make_success_response(result)


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    if not request.is_json:
        return make_error_response("Request must be JSON.", status_code=415)

    body  = request.get_json()
    texts = body.get("texts", [])

    if not isinstance(texts, list) or not texts:
        return make_error_response("'texts' must be a non-empty list.", status_code=400)
    if len(texts) > 100:
        return make_error_response("Batch limit is 100 texts.", status_code=413)
    if not model_loaded:
        return make_error_response("Model not ready.", status_code=503)

    try:
        results = predictor.predict_batch(texts)   # ✅ FIXED: use predictor
    except Exception as e:
        logger.error(f"Batch error: {e}", exc_info=True)
        return make_error_response(str(e), status_code=500)

    return make_success_response({"results": results, "count": len(results)})


@app.route("/api/examples", methods=["GET"])
def examples():
    return make_success_response({"examples": EXAMPLE_TEXTS})


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return make_error_response("Endpoint not found.", status_code=404)

@app.errorhandler(405)
def method_not_allowed(e):
    return make_error_response("Method not allowed.", status_code=405)

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Server error: {e}", exc_info=True)
    return make_error_response("Internal server error.", status_code=500)


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    logger.info(f"Starting Sentiment API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)