# src/config/settings.py
"""
Configuration settings for the Market Intelligence System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Project directories
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure required directories exist
for directory in [
    DATA_DIR,
    LOGS_DIR,
    OUTPUT_DIR,
    DATA_DIR / "raw",
    DATA_DIR / "processed",
]:
    directory.mkdir(parents=True, exist_ok=True)

# Twitter scraping configuration (no paid API)
TWITTER_CONFIG = {
    "MAX_TWEETS_PER_REQUEST": 100,
    "TARGET_HASHTAGS": [
        "#nifty50",
        "#sensex",
        "#intraday",
        "#banknifty",
        "#stockmarket",
        "#sharemarket",
        "#indianstockmarket",
        "#nse",
        "#bse",
        "#dalalstreet",
    ],
    "HOURS_BACK": 24,
    "MIN_TWEETS_TARGET": 2000,
    "REQUEST_DELAY_RANGE": (1, 5),  # seconds between requests
    "MAX_RETRIES": 3,
    "TIMEOUT": 30,  # seconds
}

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    "REQUESTS_PER_MINUTE": 30,
    "REQUESTS_PER_HOUR": 300,
    "BACKOFF_FACTOR": 2,
    "MAX_BACKOFF": 300,  # seconds
    "JITTER": True,
}

# Rotating user agents to reduce detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:117.0) Gecko/20100101 Firefox/117.0",
]

# Processing parameters
PROCESSING_CONFIG = {
    "CHUNK_SIZE": 1000,
    "MAX_WORKERS": min(8, (os.cpu_count() or 1) + 4),
    "MEMORY_LIMIT_MB": 1000,
    "BATCH_SIZE": 500,
}

# Analysis configuration
ANALYSIS_CONFIG = {
    # Use "ensemble" (VADER + TextBlob + financial lexicon) by default.
    # If you later add transformers, swap to a model name and handle it in code.
    "SENTIMENT_MODEL": "ensemble",
    "TF_IDF_MAX_FEATURES": 5000,
    "TF_IDF_MIN_DF": 2,
    "TF_IDF_MAX_DF": 0.95,
    "SIGNAL_AGGREGATION_WINDOW": "1H",
    "CONFIDENCE_THRESHOLD": 0.6,
}

# Storage configuration
STORAGE_CONFIG = {
    "PARQUET_COMPRESSION": "snappy",
    "PARQUET_ENGINE": "pyarrow",  # or "fastparquet"
    "BACKUP_RETENTION_DAYS": 30,
    "MAX_FILE_SIZE_MB": 100,
}

# Logging configuration
LOGGING_CONFIG = {
    "LEVEL": "INFO",
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "MAX_BYTES": 10 * 1024 * 1024,  # 10 MB per log file
    "BACKUP_COUNT": 5,  # keep 5 rotated files
}

# Monitoring / observability
MONITORING_CONFIG = {
    "ENABLE_PROFILING": False,
    "MEMORY_MONITORING": True,
    "PERFORMANCE_LOGGING": True,
    "METRICS_INTERVAL": 60,  # seconds
}
