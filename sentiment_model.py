# sentiment_model.py
import threading
from typing import Optional

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentModel:
    """
    Thread-safe singleton wrapper around NLTK's VADER sentiment model.

    It exposes `score_review(text) -> float` which returns a rating in [0, 5].
    """
    _instance: Optional["SentimentModel"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        # Ensure the lexicon is available (downloaded once).
        nltk.download("vader_lexicon", quiet=True)
        self._analyzer = SentimentIntensityAnalyzer()

    @classmethod
    def instance(cls) -> "SentimentModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SentimentModel()
        return cls._instance

    def score_review(self, text: str) -> float:
        """
        Convert sentiment to a 'rating' between 0 and 5.

        VADER returns a compound score in [-1, 1].
        We map it linearly to [0, 5]:
            rating = (compound + 1) * 2.5
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0

        scores = self._analyzer.polarity_scores(text)
        compound = scores.get("compound", 0.0)

        rating = (compound + 1.0) * 2.5  # -1 -> 0, 0 -> 2.5, 1 -> 5
        rating = max(0.0, min(5.0, rating))  # clamp
        return rating
