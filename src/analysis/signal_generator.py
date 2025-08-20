# src/analysis/signal_generator.py

import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from config.settings import ANALYSIS_CONFIG
from analysis.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.tfidf = TfidfVectorizer(
            max_features=ANALYSIS_CONFIG["TF_IDF_MAX_FEATURES"],
            min_df=ANALYSIS_CONFIG["TF_IDF_MIN_DF"],
            max_df=ANALYSIS_CONFIG["TF_IDF_MAX_DF"],
            ngram_range=(1, 3),
            lowercase=True,
            stop_words="english",
        )

    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["text_length"] = df["content"].astype(str).str.len()
        df["word_count"] = df["content"].astype(str).str.split().str.len()
        df["hashtag_count"] = df.get(
            "hashtags", pd.Series([[]] * len(df), index=df.index)
        ).apply(lambda x: len(x) if isinstance(x, list) else 0)
        df["mention_count"] = df.get(
            "mentions", pd.Series([[]] * len(df), index=df.index)
        ).apply(lambda x: len(x) if isinstance(x, list) else 0)
        for c in ["likes", "retweets", "replies"]:
            if c not in df.columns:
                df[c] = 0
        df["total_engagement"] = (
            df["likes"].fillna(0) + df["retweets"].fillna(0) + df["replies"].fillna(0)
        )
        df["engagement_rate"] = df["total_engagement"] / (
            df["word_count"].replace(0, np.nan).fillna(1)
        )
        return df

    def _add_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        sentiments = self.sentiment_analyzer.batch_analyze(
            df["content"].fillna("").astype(str).tolist()
        )
        s = pd.DataFrame(sentiments, index=df.index)
        df["sentiment_compound"] = s["compound"]
        df["sentiment_positive"] = s["positive"]
        df["sentiment_negative"] = s["negative"]
        df["sentiment_neutral"] = s["neutral"]
        df["sentiment_confidence"] = s["confidence"]
        return df

    def _add_tfidf(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fit TF-IDF and attach only a small set of top-importance columns (to limit memory)
        mat = self.tfidf.fit_transform(df["content"].fillna("").astype(str))
        feature_names = self.tfidf.get_feature_names_out()
        feats = pd.DataFrame(
            mat.toarray(), columns=[f"tfidf_{n}" for n in feature_names], index=df.index
        )

        # Select top N features by mean weight
        top_count = min(50, feats.shape[1]) if feats.shape[1] > 0 else 0
        if top_count > 0:
            top_cols = feats.mean().nlargest(top_count).index
            for col in top_cols:
                df[col] = feats[col]
        else:
            top_cols = []

        # Summary stats (independent of selected top columns)
        df["tfidf_mean"] = feats.mean(axis=1) if not feats.empty else 0.0
        df["tfidf_std"] = feats.std(axis=1) if not feats.empty else 0.0
        df["tfidf_max"] = feats.max(axis=1) if not feats.empty else 0.0
        df["tfidf_sum"] = feats.sum(axis=1) if not feats.empty else 0.0
        return df

    def _aggregate(self, df: pd.DataFrame, window: str) -> pd.DataFrame:
        df = df.copy()
        df = df.set_index("timestamp")
        agg: Dict[str, object] = {
            "sentiment_compound": ["mean", "std", "min", "max"],
            "sentiment_positive": "mean",
            "sentiment_negative": "mean",
            "sentiment_confidence": "mean",
            "total_engagement": "sum",
            "engagement_rate": "mean",
            "text_length": "mean",
            "word_count": "mean",
            "hashtag_count": "sum",
            "mention_count": "sum",
        }
        for col in [c for c in df.columns if c.startswith("tfidf_")]:
            agg[col] = "mean"

        res = df.resample(window).agg(agg)
        # Flatten MultiIndex columns
        res.columns = ["_".join(c) if isinstance(c, tuple) else c for c in res.columns]
        res["tweet_volume"] = df.resample(window).size()
        res = res[res["tweet_volume"] > 0].reset_index()
        return res

    def _final_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Normalize tweet volume and engagement_rate_mean
        vol_norm = (df["tweet_volume"] - df["tweet_volume"].mean()) / (
            df["tweet_volume"].std() or 1.0
        )
        eng_col = (
            "engagement_rate_mean" if "engagement_rate_mean" in df.columns else None
        )
        if eng_col:
            eng_norm = (df[eng_col] - df[eng_col].mean()) / (df[eng_col].std() or 1.0)
        else:
            eng_norm = pd.Series(0.0, index=df.index)

        vol_norm = vol_norm.fillna(0.0)
        eng_norm = eng_norm.fillna(0.0)

        volty = df.get(
            "sentiment_compound_std", pd.Series([0.0] * len(df), index=df.index)
        )
        volty_norm = -volty / (volty.max() or 1.0)

        sent_mean_col = (
            "sentiment_compound_mean"
            if "sentiment_compound_mean" in df.columns
            else None
        )
        sent_mean = (
            df[sent_mean_col] if sent_mean_col else pd.Series(0.0, index=df.index)
        )

        df["signal_score"] = (
            (0.4 * sent_mean) + (0.3 * vol_norm) + (0.2 * eng_norm) + (0.1 * volty_norm)
        )

        conf_col = (
            "sentiment_confidence_mean"
            if "sentiment_confidence_mean" in df.columns
            else None
        )
        base_conf = df[conf_col] if conf_col else pd.Series(0.5, index=df.index)
        df["confidence"] = base_conf * np.minimum(df["tweet_volume"] / 5.0, 1.0)

        def cat(x: float) -> str:
            if x >= 0.8:
                return "STRONG_BUY"
            if x >= 0.4:
                return "BUY"
            if x <= -0.8:
                return "STRONG_SELL"
            if x <= -0.4:
                return "SELL"
            return "NEUTRAL"

        df["signal"] = df["signal_score"].apply(cat)
        df["signal_strength"] = np.abs(df["signal_score"]) * df["confidence"]
        margin = 0.1
        df["signal_upper"] = df["signal_score"] + margin * (1.0 - df["confidence"])
        df["signal_lower"] = df["signal_score"] - margin * (1.0 - df["confidence"])
        return df

    def generate_signals(
        self, tweets: List[Dict], aggregation_window: str = "1H"
    ) -> pd.DataFrame:
        if not tweets:
            return pd.DataFrame()

        df = pd.DataFrame(tweets)
        if "timestamp" not in df.columns:
            df["timestamp"] = datetime.now()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        df = self._extract_text_features(df)
        df = self._add_sentiment(df)
        df = self._add_tfidf(df)
        agg = self._aggregate(df, aggregation_window)
        return self._final_signals(agg)

    def get_latest_signal(self, signals_df: pd.DataFrame) -> Optional[Dict]:
        if signals_df.empty:
            return None
        r = signals_df.iloc[-1]
        return {
            "timestamp": r.get("timestamp"),
            "signal": r.get("signal"),
            "signal_score": float(r.get("signal_score", 0.0)),
            "confidence": float(r.get("confidence", 0.0)),
            "signal_strength": float(r.get("signal_strength", 0.0)),
            "tweet_volume": int(r.get("tweet_volume", 0)),
            "sentiment_mean": float(r.get("sentiment_compound_mean", 0.0)),
        }

    def get_signal_summary(self, signals_df: pd.DataFrame) -> Dict:
        if signals_df.empty:
            return {}
        return {
            "total_periods": int(len(signals_df)),
            "signal_distribution": signals_df["signal"].value_counts().to_dict(),
            "mean_confidence": float(signals_df["confidence"].mean()),
            "mean_signal_strength": float(signals_df["signal_strength"].mean()),
            "total_tweets": int(signals_df["tweet_volume"].sum()),
            "avg_sentiment": float(
                signals_df.get("sentiment_compound_mean", pd.Series([0.0])).mean()
            ),
            "sentiment_volatility": float(
                signals_df.get("sentiment_compound_std", pd.Series([0.0])).mean()
            ),
        }
