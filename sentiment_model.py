# sentiment_model.py
"""
SentimentModel using HuggingFace:
    nlptown/bert-base-multilingual-uncased-sentiment

This model predicts star ratings from 1 to 5 directly from text reviews.
We convert that into a float rating in [1, 5] by taking the expected value
of the star distribution:

    rating = sum_{s=1..5} p(s) * s

So:
  - Very negative → close to 1
  - Neutral       → around 3
  - Very positive → close to 5

Your existing code just needs:
    from sentiment_model import SentimentModel
    score = SentimentModel.instance().score_review(text)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"


@dataclass
class SentimentModel:
    """
    Singleton wrapper around the HuggingFace star-rating model.

    Usage:
        model = SentimentModel.instance()
        rating = model.score_review("The food was amazing!")
    """

    _instance: ClassVar["SentimentModel | None"] = None

    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: int  # -1 = CPU, 0 = cuda:0, etc.

    def __init__(self) -> None:
        # Load tokenizer + model once
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

        # Choose device (GPU if available, otherwise CPU)
        if torch.cuda.is_available():
            self.device = 0
            self.model.to(self.device)
        else:
            self.device = -1  # CPU

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "SentimentModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Text -> rating / 5
    # ------------------------------------------------------------------
    def score_review(self, text: str) -> float:
        """
        Analyze a review text and return a rating in [1, 5].

        Steps:
          1. Tokenize text for the nlptown model (truncating long reviews).
          2. Get logits for the 5 star classes.
          3. Softmax -> probabilities p_1..p_5.
          4. Compute expected stars: rating = sum(s * p_s), s in {1..5}.
          5. Round to 2 decimals.

        If the text is empty, we return 3.0 (neutral).
        """
        if not text or not text.strip():
            return 3.0  # neutral 3/5

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        if self.device >= 0:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass (no gradients needed)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape [1, 5]

        # Convert logits -> probabilities via softmax
        probs = F.softmax(logits, dim=-1)[0]  # shape [5]

        # Expected star value in [1, 5]
        rating = 0.0
        for i in range(5):
            stars = i + 1  # labels are 1..5 stars
            rating += stars * float(probs[i])

        # Round for stability
        return round(float(rating), 2)
