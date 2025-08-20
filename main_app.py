# main_app.py

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Dict

# Ensure src is on sys.path (use double underscores in __file__)
# If you still face issues with imports, you can replace the next line with:
# sys.path.append(r"F:\EPAT\market-intelligence-system\src")
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from utils.helpers import setup_logging, performance_monitor  # noqa: E402
from config.settings import TWITTER_CONFIG, ANALYSIS_CONFIG  # noqa: E402
from data_collection.twitter_scraper import TwitterScraper  # noqa: E402
from data_processing.text_cleaner import TextCleaner  # noqa: E402
from data_processing.data_storage import DataStorage  # noqa: E402
from analysis.signal_generator import SignalGenerator  # noqa: E402
from visualization.memory_efficient_plots import MemoryEfficientPlotter  # noqa: E402


class MarketIntelligenceSystem:
    def __init__(self):
        self.logger = setup_logging()
        self.scraper = TwitterScraper()
        self.cleaner = TextCleaner()
        self.storage = DataStorage()
        self.signal_gen = SignalGenerator()
        self.plotter = MemoryEfficientPlotter()
        self.stats = {}

    def run_full_pipeline(
        self, hashtags: List[str], max_tweets: int, hours_back: int
    ) -> Dict:
        self.logger.info("Starting pipeline")

        # 1) Collect tweets
        with performance_monitor("tweet_collection"):
            tweets = self.scraper.scrape_tweets(
                hashtags=hashtags, max_tweets=max_tweets, hours_back=hours_back
            )
        tweet_dicts = [t.to_dict() for t in tweets]
        self.stats["tweets_collected"] = len(tweet_dicts)
        self.logger.info(f"Collected {self.stats['tweets_collected']} tweets")

        # 2) Clean + enrich
        texts = [t.get("content", "") for t in tweet_dicts]
        cleaned = self.cleaner.batch_clean(texts)
        for i, tw in enumerate(tweet_dicts):
            tw["cleaned_content"] = cleaned[i]
        tweet_file = self.storage.save_tweets(tweet_dicts)

        # 3) Generate signals
        signals_df = self.signal_gen.generate_signals(
            tweet_dicts, aggregation_window=ANALYSIS_CONFIG["SIGNAL_AGGREGATION_WINDOW"]
        )
        signals_file = self.storage.save_signals(signals_df)

        # 4) Dashboard output
        dash_path = (
            self.storage.output_path
            / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        fig = self.plotter.plot_sentiment_timeline(signals_df)
        fig.write_html(str(dash_path))

        latest = self.signal_gen.get_latest_signal(signals_df)
        summary = self.signal_gen.get_signal_summary(signals_df)

        result = {
            "status": "success",
            "statistics": {"tweets_collected": self.stats["tweets_collected"]},
            "files": {
                "tweets": tweet_file,
                "signals": signals_file,
                "dashboard": str(dash_path),
            },
            "latest_signal": latest,
            "summary": summary,
        }
        self.logger.info("Pipeline completed successfully")
        return result


def main():
    parser = argparse.ArgumentParser(description="Real-Time Market Intelligence System")
    parser.add_argument(
        "--hashtags", nargs="+", default=TWITTER_CONFIG["TARGET_HASHTAGS"]
    )
    parser.add_argument(
        "--max-tweets", type=int, default=TWITTER_CONFIG["MIN_TWEETS_TARGET"]
    )
    parser.add_argument("--hours-back", type=int, default=TWITTER_CONFIG["HOURS_BACK"])
    args = parser.parse_args()

    system = MarketIntelligenceSystem()
    result = system.run_full_pipeline(args.hashtags, args.max_tweets, args.hours_back)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
