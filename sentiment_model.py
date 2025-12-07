# sentiment_model.py
"""
SentimentModel using VADER (lexicon-based sentiment analysis).

We previously used the HuggingFace model:
    nlptown/bert-base-multilingual-uncased-sentiment
but that was too slow on CPU for thousands of reviews.

Now we switch to VADER, which is:
  - very lightweight
  - fast even for tens of thousands of reviews
  - good enough for demo / academic purposes

VADER returns a "compound" sentiment score in [-1, 1].
We convert this to a rating in [0, 5] as:

    rating = (compound + 1) / 2 * 5

So:
  - compound = -1 → rating = 0
  - compound = 0  → rating = 2.5
  - compound = +1 → rating = 5

Interface (unchanged from the HuggingFace version):

    from sentiment_model import SentimentModel

    model = SentimentModel.instance()
    single_rating = model.score_review("The food was amazing!")
    batch_ratings = model.score_reviews_batch(["Good", "Bad", "Meh"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable, List

import math

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentModel:
    """
    Singleton wrapper around VADER sentiment analyzer.

    Usage:
        model = SentimentModel.instance()
        rating = model.score_review("The food was amazing!")
        ratings = model.score_reviews_batch(["Good", "Bad"])
    """

    _instance: ClassVar["SentimentModel | None"] = None

    analyzer: SentimentIntensityAnalyzer

    def __init__(self) -> None:
        # Ensure the VADER lexicon is available
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            # Download once if missing – you can also do this manually
            nltk.download("vader_lexicon")

        self.analyzer = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "SentimentModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Single-review helper (uses batch implementation)
    # ------------------------------------------------------------------
    def score_review(self, text: str) -> float:
        """
        Analyze a review text and return a rating in [0, 5].

        If the text is empty, we return 2.5 (neutral).
        """
        if not text or not text.strip():
            return 2.5
        ratings = self.score_reviews_batch([text])
        return ratings[0] if ratings else 2.5

    # ------------------------------------------------------------------
    # Batch scoring – same API as the HuggingFace version
    # ------------------------------------------------------------------
    def score_reviews_batch(self, texts: Iterable[str]) -> List[float]:
        """
        Analyze a batch of review texts and return ratings in [0, 5].

        For each text:
          1. Use VADER to get 'compound' score in [-1, 1].
          2. Map to [0, 5]:
                rating = (compound + 1) / 2 * 5
          3. Clamp to [0, 5] and round to 2 decimals.

        Empty / whitespace-only texts are treated as neutral 2.5.
        """
        results: List[float] = []

        for t in texts:
            if not t or not str(t).strip():
                results.append(2.5)
                continue

            scores = self.analyzer.polarity_scores(str(t))
            compound = scores.get("compound", 0.0)

            # Map [-1, 1] -> [0, 5]
            rating = (compound + 1.0) / 2.0 * 5.0

            # Clamp to [0, 5]
            rating = max(0.0, min(5.0, rating))

            # Round to 2 decimals
            rating = round(rating, 2)

            results.append(rating)

        return results

