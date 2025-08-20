# src/visualization/memory_efficient_plots.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MemoryEfficientPlotter:
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= self.max_points:
            return df
        step = max(1, len(df) // self.max_points)
        return df.iloc[::step].copy()

    def plot_sentiment_timeline(self, signals_df: pd.DataFrame):
        if signals_df is None or signals_df.empty:
            return go.Figure()

        df = self._sample(signals_df)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("Sentiment", "Signal Strength", "Tweet Volume"),
        )

        # Sentiment with confidence band
        if "sentiment_compound_mean" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["sentiment_compound_mean"],
                    mode="lines+markers",
                    name="Sentiment",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )
            if "sentiment_compound_std" in df.columns:
                upper = df["sentiment_compound_mean"] + df["sentiment_compound_std"]
                lower = df["sentiment_compound_mean"] - df["sentiment_compound_std"]
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=upper,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=lower,
                        fill="tonexty",
                        mode="lines",
                        name="Confidence Band",
                        line=dict(width=0),
                        fillcolor="rgba(0,100,80,0.2)",
                    ),
                    row=1,
                    col=1,
                )

        # Signal strength
        if "signal_strength" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["signal_strength"],
                    mode="lines",
                    name="Signal Strength",
                    line=dict(color="green"),
                ),
                row=2,
                col=1,
            )

        # Tweet volume
        if "tweet_volume" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df["timestamp"],
                    y=df["tweet_volume"],
                    name="Tweet Volume",
                    marker_color="orange",
                ),
                row=3,
                col=1,
            )

        fig.update_layout(
            height=800,
            template="plotly_white",
            title="Market Sentiment Analysis",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig
