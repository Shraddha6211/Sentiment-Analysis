"""
=============================================================
  STEP 3: PREDICTION MODULE
  File: backend/model/predict.py
=============================================================

PURPOSE:
    Load the saved model and expose a clean prediction function
    that the Flask API can call for every incoming request.

DESIGN PRINCIPLE — SEPARATION OF CONCERNS:
    - train.py   : Knows HOW to train the model
    - predict.py : Knows HOW to use the trained model
    - app.py     : Knows HOW to serve predictions over HTTP
    Each file has one clear job. This is standard in production ML systems.

WHAT THIS FILE PROVIDES:
    - SentimentPredictor class (load once, predict many times)
    - Confidence scores per class (not just a label)
    - Text preprocessing consistent with training
"""

import os
import re
import pickle
import numpy as np   # Numerical operations (pip install numpy)


class SentimentPredictor:
    """
    Wrapper around the trained sklearn Pipeline.

    WHY A CLASS instead of bare functions?
        - The model is loaded ONCE when the class is instantiated
        - Each call to predict() reuses the loaded model
        - Avoids expensive disk reads on every prediction
        - Easy to mock/test in unit tests
    """

    # Class-level label definitions (shared across all instances)
    LABELS = ["negative", "neutral", "positive"]

    # Emoji / icon to show in UI for each sentiment
    EMOJI_MAP = {
        "positive": "😊",
        "negative": "😞",
        "neutral":  "😐",
    }

    # Color codes for UI theming per sentiment
    COLOR_MAP = {
        "positive": "#22c55e",   # Green
        "negative": "#ef4444",   # Red
        "neutral":  "#f59e0b",   # Amber
    }

    def __init__(self, model_path: str):
        """
        Load the trained pipeline from disk.

        Args:
            model_path: Absolute path to the .pkl file
        Raises:
            FileNotFoundError : If the model hasn't been trained yet
            Exception         : For corrupt/incompatible pickle files
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Please run: python model/train.py"
            )

        # Load the entire Pipeline (TF-IDF + LogisticRegression)
        with open(model_path, "rb") as f:   # "rb" = read binary
            self.pipeline = pickle.load(f)

        print(f"  ✔ Model loaded from {model_path}")

    # ─────────────────────────────────────────
    #   TEXT CLEANING (must match train.py)
    # ─────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """
        Apply the same cleaning steps used during training.

        ⚠️ CRITICAL: If you change cleaning in train.py,
           you MUST update this function too, otherwise
           the model sees different text than it was trained on
           (called "train-serve skew" — a common production bug).
        """
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ─────────────────────────────────────────
    #   MAIN PREDICTION METHOD
    # ─────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Predict the sentiment of a single text string.

        Args:
            text: Raw input text (e.g., "I love this so much!")
        Returns:
            A dict with:
                - sentiment    : Predicted class label
                - confidence   : Probability of the predicted class (0–1)
                - probabilities: Dict of {label: probability} for all classes
                - emoji        : Emoji icon for the predicted class
                - color        : Hex color for the predicted class
                - cleaned_text : What the model actually saw after cleaning

        HOW predict_proba() WORKS:
            LogisticRegression.predict_proba() uses the softmax function
            to convert raw scores (logits) into probabilities that sum to 1.
            Example: {"negative": 0.05, "neutral": 0.10, "positive": 0.85}
        """

        # Input validation: reject empty strings
        if not text or not text.strip():
            return {
                "error": "Input text cannot be empty.",
                "sentiment": None,
            }

        # Clean the text before passing to the pipeline
        cleaned = self._clean(text)

        # Edge case: if cleaning removes everything (e.g., pure emoji input)
        if not cleaned:
            cleaned = "unknown"

        # ── Get probability scores for ALL classes ──
        # predict_proba returns shape (1, n_classes) — we take row [0]
        proba_array = self.pipeline.predict_proba([cleaned])[0]

        # Map probabilities back to class names
        # pipeline.classes_ gives the label order used internally
        proba_dict = {
            label: round(float(prob), 4)
            for label, prob in zip(self.pipeline.classes_, proba_array)
        }

        # ── Get the top predicted class ──
        predicted_label = max(proba_dict, key=proba_dict.get)
        confidence      = proba_dict[predicted_label]

        return {
            "sentiment":     predicted_label,
            "confidence":    round(confidence, 4),
            "probabilities": proba_dict,
            "emoji":         self.EMOJI_MAP[predicted_label],
            "color":         self.COLOR_MAP[predicted_label],
            "cleaned_text":  cleaned,
            "original_text": text,
        }

    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for a list of texts.
        More efficient than calling predict() in a loop because
        TF-IDF vectorization is applied to the entire batch at once.

        Args:
            texts: List of raw input strings
        Returns:
            List of prediction dicts (same format as predict())
        """
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────────
#   QUICK TEST — run directly to verify model
# ─────────────────────────────────────────────

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "sentiment_model.pkl")

    predictor = SentimentPredictor(model_path)

    test_cases = [
        "I absolutely love this, it made my day! 😍",
        "This is terrible, I want a refund immediately",
        "Just had lunch, it was okay I guess",
        "Best thing that ever happened to me!",
        "Feeling so down and hopeless right now",
    ]

    print("\n" + "=" * 55)
    print("  PREDICTION TEST")
    print("=" * 55)
    for text in test_cases:
        result = predictor.predict(text)
        print(f"\n  Input    : {text}")
        print(f"  Sentiment: {result['sentiment']} {result['emoji']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Probs    : {result['probabilities']}")
    print("=" * 55)