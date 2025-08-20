# Real-Time Market Intelligence System

A production-ready Python project for scraping, analyzing, and visualizing Indian stock market sentiment in real time from Twitter/X—without paid APIs.

---

## 📌 Features

- **No paid Twitter API required:** Uses Selenium for live scraping with robust anti-detection.
- **Multi-hashtag tracking:** Easily configure target hashtags like `#nifty50`, `#sensex`, `#banknifty`.
- **Comprehensive data:** Collects tweet content, user info, engagement, hashtags, mentions, timestamps.
- **Data engineering:** Cleans, deduplicates, and stores data in Parquet for speed and memory efficiency.
- **Advanced NLP:** Sentiment analysis (ensemble: VADER, TextBlob, market lexicon), text features, TF-IDF.
- **Quant trading signals:** Aggregates features into actionable signals with configurable confidence.
- **Memory-efficient visualization:** Interactive dashboards and streaming plots with Plotly.
- **Modern Python stack:** Modular, fully documented, production logging, error handling, test suite.
- **Easy to run (local or cloud):** One command to launch the full pipeline.

---

## 🚦 Quick Start

**Requirements:**  
- Python 3.9+  
- Google Chrome (latest)  
- Chrome driver auto-installs

Clone the repo
git clone https://github.com/vinaymanchanda/market-intelligence-system.git
cd market-intelligence-system

Set up a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

Or on Mac/Linux
python3 -m venv venv && source venv/bin/activate
Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

Install NLTK datasets (first run only, at Python prompt)
python

import nltk
nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('vader_lexicon')
exit()

Run the pipeline (collect 200 tweets from #nifty50 over 24hr)
python main_app.py --hashtags "#nifty50" --max-tweets 200 --hours-back 24


---

## 🗂 Project Structure

market-intelligence-system/
main_app.py # Entry point
requirements.txt
src/
config/ # Settings
utils/ # Helpers; logging
data_collection/ # Selenium scraper, ratelimiter
data_processing/ # Cleaner, storage
analysis/ # Sentiment, signal generator
visualization/ # Dashboard/plotly
data/
raw/ # Parquet raw data
processed/ # Clean data
output/ # Results, dashboards
logs/ # All logs (rotated)
tests/ # Unit tests

text

---

## 🔥 What it Does

- **Scraping:** Opens live Twitter search, scrolls, parses tweets for top Indian market hashtags
- **Processing:** Cleans unicode, handles emojis, removes dups, robust to language noise
- **Analysis:** Per-tweet and aggregated sentiment, TF-IDF, text feature engineering
- **Signal Generation:** Converts all features into time-windowed buy/sell/neutral trading signals
- **Visualization:** Plotly dashboards for trends, volatility, signal confidence, traffic

---

## 🏗 Configuration

All settings in `src/config/settings.py`:

- **Change hashtags/volume:**  
  `TWITTER_CONFIG["TARGET_HASHTAGS"]`  
  `TWITTER_CONFIG["MIN_TWEETS_TARGET"]`

- **Storage:**  
  Parquet with snappy compression

- **Analysis:**  
  TF-IDF max features, aggregation window, confidence threshold

- **Browser profile:**  
  Use persistent Chrome profile to avoid X login walls

---

## 🚨 Troubleshooting

- **Selenium “cannot find Chrome binary”:**  
  Install Google Chrome, set `options.binary_location` if needed.

- **NLTK “resource not found”:**  
  Run the `nltk.download(...)` lines above once.

- **Twitter blocks or consent screen:**  
  Run non-headless, log in, and use a persistent browser profile as documented in code comments.

- **Import errors:**  
  Use `sys.path.append(...)` in main_app.py as shown, or run as `python -m main_app ...` from project root.

- **Timeouts:**  
  Check Internet, user agent, and try increasing timeouts in scraper.

---

## 🧪 Tests

pytest

text

---

## 🙏 Credits

- Inspired by real production quant infrastructure.
- Market lexicon/tags are customizable for your research.

---

## 📜 License

MIT License

---

