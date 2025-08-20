# src/analysis/sentiment_analyzer.py

import logging
from typing import List, Dict
import numpy as np  # reserved for future extensions
import pandas as pd  # reserved for future extensions
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import re

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self):
        # Try to initialize VADER; if resources missing, fall back gracefully
        try:
            self.vader = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(
                f"VADER initialization failed: {e}. Falling back without VADER."
            )
            self.vader = None
        self.financial_keywords = self._keywords()

    def _keywords(self) -> Dict[str, float]:
        pos = {
            "bullish": 2.0,
            "bull": 1.5,
            "rally": 2.0,
            "surge": 2.0,
            "boom": 2.0,
            "green": 1.0,
            "profit": 1.5,
            "gain": 1.5,
            "up": 1.0,
            "high": 1.0,
            "strong": 1.5,
            "positive": 1.0,
            "buy": 1.5,
            "long": 1.0,
            "support": 1.0,
            "breakout": 2.0,
            "momentum": 1.5,
            "growth": 1.5,
            "rise": 1.5,
            "climb": 1.5,
        }
        neg = {
            "bearish": -2.0,
            "bear": -1.5,
            "crash": -2.5,
            "dump": -2.0,
            "fall": -1.5,
            "red": -1.0,
            "loss": -1.5,
            "down": -1.0,
            "low": -1.0,
            "weak": -1.5,
            "negative": -1.0,
            "sell": -1.5,
            "short": -1.0,
            "resistance": -1.0,
            "breakdown": -2.0,
            "decline": -1.5,
            "drop": -1.5,
            "plunge": -2.0,
        }
        return {**pos, **neg}

    def _vader_sentiment(self, text: str) -> Dict:
        if not self.vader:
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "confidence": 0.0,
                "method": "vader",
            }
        s = self.vader.polarity_scores(text or "")
        compound = float(s.get("compound", 0.0))
        return {
            "compound": compound,
            "positive": float(s.get("pos", 0.0)),
            "negative": float(s.get("neg", 0.0)),
            "neutral": float(s.get("neu", 1.0)),
            "confidence": abs(compound),
            "method": "vader",
        }

    def _textblob_sentiment(self, text: str) -> Dict:
        blob = TextBlob(text or "")
        pol = float(getattr(blob.sentiment, "polarity", 0.0))
        return {
            "compound": pol,
            "positive": max(0.0, pol),
            "negative": max(0.0, -pol),
            "neutral": 1.0 - abs(pol),
            "confidence": abs(pol),
            "method": "textblob",
        }

    def _financial_sentiment(self, text: str) -> Dict:
        words = re.findall(r"\b\w+\b", (text or "").lower())
        score = 0.0
        matched = 0
        for w in words:
            if w in self.financial_keywords:
                score += self.financial_keywords[w]
                matched += 1
        comp = 0.0 if matched == 0 else max(-1.0, min(1.0, (score / matched) / 2.0))
        return {
            "compound": comp,
            "positive": max(0.0, comp),
            "negative": max(0.0, -comp),
            "neutral": 1.0 - abs(comp),
            "confidence": min(1.0, matched / 5.0),
            "method": "financial",
        }

    def analyze_sentiment(self, text: str, method: str = "ensemble") -> Dict:
        if not (text or "").strip():
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "confidence": 0.0,
                "method": "empty",
            }

        if method == "vader":
            return self._vader_sentiment(text)
        if method == "textblob":
            return self._textblob_sentiment(text)
        if method == "financial":
            return self._financial_sentiment(text)

        v = self._vader_sentiment(text)
        t = self._textblob_sentiment(text)
        f = self._financial_sentiment(text)

        comp = 0.3 * v["compound"] + 0.2 * t["compound"] + 0.5 * f["compound"]
        conf = 0.3 * v["confidence"] + 0.2 * t["confidence"] + 0.5 * f["confidence"]

        return {
            "compound": comp,
            "positive": max(0.0, comp),
            "negative": max(0.0, -comp),
            "neutral": 1.0 - abs(comp),
            "confidence": conf,
            "method": "ensemble",
            "components": {"vader": v, "textblob": t, "financial": f},
        }

    def batch_analyze(self, texts: List[str], method: str = "ensemble") -> List[Dict]:
        return [self.analyze_sentiment(x, method) for x in texts]
