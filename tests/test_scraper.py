import unittest
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from data_collection.rate_limiter import RateLimiter
from analysis.sentiment_analyzer import SentimentAnalyzer
from data_processing.text_cleaner import TextCleaner


class TestBasics(unittest.TestCase):
    def test_rate_limiter(self):
        rl = RateLimiter()
        rl.record_failure()
        rl.record_success()
        stats = rl.get_stats()
        self.assertIn("total_requests", stats)
        self.assertIn("failure_count", stats)

    def test_sentiment(self):
        sa = SentimentAnalyzer()
        pos = sa.analyze_sentiment("Massive bullish breakout on #nifty50!")
        neg = sa.analyze_sentiment("Huge crash incoming. Bearish tone.")
        self.assertGreater(pos["compound"], neg["compound"])

    def test_cleaner(self):
        cl = TextCleaner()
        out = cl.clean_tweet("Hello @user visit https://x.com #nifty50 🚀")
        self.assertIn("#nifty50", out)
        self.assertNotIn("https://", out)


if __name__ == "__main__":
    unittest.main()
