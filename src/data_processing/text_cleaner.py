# src/data_processing/text_cleaner.py

import re
import unicodedata
from typing import List, Dict
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import emoji

logger = logging.getLogger(__name__)


class TextCleaner:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception as e:
            logger.warning(
                f"NLTK stopwords not available: {e}. Proceeding without stopwords."
            )
            self.stop_words = set()

        self.market_stop_words = {
            "stock",
            "market",
            "trading",
            "buy",
            "sell",
            "hold",
            "price",
            "target",
            "support",
            "resistance",
        }

        self.url_pattern = re.compile(r"http[s]?://\S+")
        self.mention_pattern = re.compile(r"@[A-Za-z0-9_]+")
        self.hashtag_pattern = re.compile(r"#[A-Za-z0-9_]+")
        self.whitespace_pattern = re.compile(r"\s+")

    def clean_tweet(
        self, text: str, preserve_hashtags: bool = True, preserve_mentions: bool = False
    ) -> str:
        if not text:
            return ""

        # Preserve tags (optionally)
        hashtags = self.hashtag_pattern.findall(text) if preserve_hashtags else []
        mentions = self.mention_pattern.findall(text) if preserve_mentions else []

        # Remove URLs and normalize
        text = self.url_pattern.sub("", text)
        try:
            text = emoji.demojize(text, delimiters=(" ", " "))
        except Exception:
            pass
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch if ch.isprintable() else " " for ch in text)

        # Remove mentions/hashtags if not preserving
        if not preserve_mentions:
            text = self.mention_pattern.sub("", text)
        if not preserve_hashtags:
            text = self.hashtag_pattern.sub("", text)

        # Collapse whitespace
        text = self.whitespace_pattern.sub(" ", text).strip()

        # Tokenize
        try:
            tokens = word_tokenize(text.lower())
        except Exception:
            tokens = text.lower().split()

        # Filter and lemmatize
        cleaned = []
        for tok in tokens:
            if tok in self.stop_words or tok in self.market_stop_words:
                continue
            if tok.isdigit() or len(tok) < 2 or len(tok) > 20:
                continue
            try:
                tok = self.lemmatizer.lemmatize(tok)
            except Exception:
                pass
            if tok.isalpha():
                cleaned.append(tok)

        out = " ".join(cleaned)

        # Re-attach preserved tags at the end
        if preserve_hashtags and hashtags:
            out = (out + " " + " ".join(hashtags)).strip()
        if preserve_mentions and mentions:
            out = (out + " " + " ".join(mentions)).strip()
        return out

    def extract_features(self, text: str) -> Dict:
        blob = TextBlob(text or "")
        return {
            "length": len(text or ""),
            "word_count": len((text or "").split()),
            "hashtag_count": len(self.hashtag_pattern.findall(text or "")),
            "mention_count": len(self.mention_pattern.findall(text or "")),
            "exclamation_count": (text or "").count("!"),
            "question_count": (text or "").count("?"),
            "polarity": float(getattr(blob.sentiment, "polarity", 0.0)),
            "subjectivity": float(getattr(blob.sentiment, "subjectivity", 0.0)),
        }

    def batch_clean(self, texts: List[str], **kwargs) -> List[str]:
        return [self.clean_tweet(t, **kwargs) for t in texts]
