# src/data_processing/data_storage.py

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from config.settings import STORAGE_CONFIG, DATA_DIR

logger = logging.getLogger(__name__)


class DataStorage:
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or DATA_DIR
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.output_path = self.base_path / "output"
        for p in [self.raw_path, self.processed_path, self.output_path]:
            p.mkdir(parents=True, exist_ok=True)
        self.compression = STORAGE_CONFIG["PARQUET_COMPRESSION"]
        self.engine = STORAGE_CONFIG["PARQUET_ENGINE"]

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            try:
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                elif pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast="float")
                elif pd.api.types.is_object_dtype(df[col]):
                    # Convert low-cardinality object columns to category
                    nunique = df[col].nunique(dropna=True)
                    if len(df) > 0 and (nunique / len(df)) < 0.5:
                        df[col] = df[col].astype("category")
            except Exception:
                # Keep original dtype on failure
                pass
        return df

    def save_tweets(self, tweets: List[Dict], filename: Optional[str] = None) -> str:
        if not tweets:
            raise ValueError("No tweets to save")
        if filename is None:
            filename = f"tweets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        path = self.raw_path / filename

        df = pd.DataFrame(tweets)
        df = self._optimize_dtypes(df)
        df.to_parquet(
            path, compression=self.compression, engine=self.engine, index=False
        )

        logger.info(f"Saved {len(df)} tweets to {path}")
        return str(path)

    def save_signals(
        self, signals_df: pd.DataFrame, filename: Optional[str] = None
    ) -> str:
        if signals_df is None or signals_df.empty:
            raise ValueError("No signals to save")
        if filename is None:
            filename = f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        path = self.output_path / filename

        df = self._optimize_dtypes(signals_df)
        df.to_parquet(
            path, compression=self.compression, engine=self.engine, index=False
        )

        logger.info(f"Saved signals to {path}")
        return str(path)
