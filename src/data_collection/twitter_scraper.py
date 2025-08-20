# src/data_collection/twitter_scraper.py
"""
Twitter/X Scraper for Market Intelligence
Selenium-based scraping (no paid APIs), with hardened headless and fallback tactics.
"""

import time
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import hashlib
import re
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent

from utils.exceptions import ScrapingError
from config.settings import TWITTER_CONFIG, USER_AGENTS
from data_collection.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class Tweet:
    id: str
    username: str
    content: str
    timestamp: datetime
    likes: int
    retweets: int
    replies: int
    hashtags: List[str]
    mentions: List[str]
    url: str

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "username": self.username,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "likes": self.likes,
            "retweets": self.retweets,
            "replies": self.replies,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "url": self.url,
        }


class TwitterScraper:
    def __init__(self, headless: bool = False, use_persistent_profile: bool = True):
        self.rate_limiter = RateLimiter()
        self.user_agent = UserAgent()
        self.driver: Optional[webdriver.Chrome] = None
        self.headless = headless
        self.use_persistent_profile = use_persistent_profile

    def _setup_driver(self) -> webdriver.Chrome:
        options = Options()

        # Toggle headless. Start non-headless for debugging; switch to headless once stable.
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1400,900")

        # Stability flags for headless + Windows GPU issues
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option(
            "excludeSwitches", ["enable-automation", "enable-logging"]
        )
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument(
            "--disable-features=IsolateOrigins,site-per-process,TranslateUI"
        )
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-webgl")
        options.add_argument(
            "--enable-unsafe-swiftshader"
        )  # address WebGL fallback warning
        options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")

        # Optional: persist a Chrome profile to keep cookies (useful if you log into X once manually)
        if self.use_persistent_profile:
            # Adjust to a path you can write to
            options.add_argument(
                r"--user-data-dir=C:\Users\vmanc\AppData\Local\Google\Chrome\User Data\SeleniumProfile"
            )
            options.add_argument("--profile-directory=Default")

        # Optional: If Chrome is not in PATH, set binary_location explicitly
        # options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(60)
        driver.set_script_timeout(60)
        driver.implicitly_wait(5)

        # Hide webdriver flag
        try:
            driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
        except Exception:
            pass

        return driver

    def _extract_metric(self, element, metric_type: str) -> int:
        # metric_type in {"like","retweet","reply"}
        selectors = [
            f"[data-testid='{metric_type}'] span",
            f"div[data-testid='{metric_type}'] span",
        ]
        for sel in selectors:
            try:
                metric_elem = element.find_element(By.CSS_SELECTOR, sel)
                metric_text = (metric_elem.text or "").strip()
                if not metric_text:
                    continue
                if "K" in metric_text:
                    return int(float(metric_text.replace("K", "")) * 1_000)
                if "M" in metric_text:
                    return int(float(metric_text.replace("M", "")) * 1_000_000)
                if metric_text.isdigit():
                    return int(metric_text)
            except Exception:
                continue
        return 0

    def _extract_tweet(self, el) -> Optional[Tweet]:
        try:
            username_el = None
            content_el = None

            # Multiple locators for username
            for sel in [
                "[data-testid='User-Name'] span",
                "div[dir='ltr'] span",
                "a[role='link'] span",
            ]:
                try:
                    cand = el.find_element(By.CSS_SELECTOR, sel)
                    if cand and cand.text.strip():
                        username_el = cand
                        break
                except Exception:
                    continue

            # Multiple locators for content
            for sel in ["[data-testid='tweetText']", "div[lang]", "article div[lang]"]:
                try:
                    cand = el.find_element(By.CSS_SELECTOR, sel)
                    if cand and cand.text.strip():
                        content_el = cand
                        break
                except Exception:
                    continue

            username = username_el.text.strip() if username_el else "unknown"
            content = content_el.text.strip() if content_el else ""

            likes = self._extract_metric(el, "like")
            retweets = self._extract_metric(el, "retweet")
            replies = self._extract_metric(el, "reply")

            hashtags = re.findall(r"#\w+", content or "")
            mentions = re.findall(r"@\w+", content or "")

            # Create a deterministic-ish id keyed to minute to reduce duplicates
            base = f"{username}_{(content or '')[:100]}_{int(time.time())//60}"
            tweet_id = hashlib.md5(base.encode("utf-8")).hexdigest()
            url = f"https://twitter.com/{username}/status/{tweet_id}"

            return Tweet(
                id=tweet_id,
                username=username,
                content=content,
                timestamp=datetime.now(),
                likes=likes,
                retweets=retweets,
                replies=replies,
                hashtags=hashtags,
                mentions=mentions,
                url=url,
            )
        except Exception as e:
            logger.debug(f"Tweet extraction failed: {e}")
            return None

    def _is_recent(self, tweet: Tweet, hours_back: int) -> bool:
        cutoff = datetime.now() - timedelta(hours=hours_back)
        return tweet.timestamp >= cutoff

    def _dedup(self, tweets: List[Tweet]) -> List[Tweet]:
        seen = set()
        out: List[Tweet] = []
        for t in tweets:
            key = re.sub(r"\s+", " ", (t.content or "").lower().strip())
            h = hashlib.md5(key.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                out.append(t)
        logger.info(f"Removed {len(tweets) - len(out)} duplicate tweets")
        return out

    def _dismiss_banners(self):
        # Try dismissing consent dialogs if present
        xpaths = [
            "//span[contains(., 'Accept')]",
            "//span[contains(., 'Allow all')]",
            "//span[contains(., 'allow all')]",
            "//button[contains(., 'Accept')]",
            "//button[contains(., 'Allow all')]",
        ]
        for xp in xpaths:
            try:
                elems = WebDriverWait(self.driver, 3).until(
                    EC.presence_of_all_elements_located((By.XPATH, xp))
                )
                for e in elems:
                    try:
                        e.click()
                        time.sleep(1)
                        return
                    except Exception:
                        continue
            except Exception:
                continue

    def _scrape_hashtag(self, hashtag: str, limit: int, hours_back: int) -> List[Tweet]:
        collected: List[Tweet] = []
        q = f"{hashtag} -filter:retweets"
        # Mobile-lite param sometimes reduces friction
        url = f"https://twitter.com/search?q={q}&src=typed_query&f=live&pf=on"

        self.driver.get(url)
        logger.info(f"Loaded URL for {hashtag}: {self.driver.current_url}")
        logger.info(f"Page title: {self.driver.title}")

        # Dismiss cookie/consent banners if any
        self._dismiss_banners()

        # Pre-scroll to trigger dynamic content
        for _ in range(2):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(random.uniform(*TWITTER_CONFIG["REQUEST_DELAY_RANGE"]))

        # Wait for any tweet-like element
        selectors = [
            "[data-testid='tweet']",
            "article div[data-testid='tweetText']",
            "article [data-testid='User-Name']",
            "article div[lang]",
        ]

        found = False
        for sel in selectors:
            try:
                WebDriverWait(self.driver, 25).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                )
                found = True
                break
            except Exception:
                continue

        if not found:
            logger.warning(f"No tweet cards found for {hashtag} after initial wait.")
            return collected

        last_height = self.driver.execute_script("return document.body.scrollHeight")
        stagnant_rounds = 0

        while len(collected) < limit and stagnant_rounds < 3:
            cards = []
            for sel in selectors:
                try:
                    cards = self.driver.find_elements(By.CSS_SELECTOR, sel)
                    if cards:
                        break
                except Exception:
                    continue

            start_len = len(collected)
            for el in cards:
                if len(collected) >= limit:
                    break
                tw = self._extract_tweet(el)
                if tw and self._is_recent(tw, hours_back):
                    collected.append(tw)

            # Scroll to load more content
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(random.uniform(*TWITTER_CONFIG["REQUEST_DELAY_RANGE"]))
            new_height = self.driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height and len(collected) == start_len:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            last_height = new_height

        return collected

    def scrape_tweets(
        self, hashtags: List[str], max_tweets: int = 2000, hours_back: int = 24
    ) -> List[Tweet]:
        logger.info(f"Starting scrape for {hashtags}")
        all_tweets: List[Tweet] = []
        self.driver = self._setup_driver()
        try:
            per_hashtag = max(1, max_tweets // max(1, len(hashtags)))
            for tag in hashtags:
                self.rate_limiter.wait_if_needed()
                batch = self._scrape_hashtag(tag, per_hashtag, hours_back)
                all_tweets.extend(batch)
                time.sleep(random.uniform(*TWITTER_CONFIG["REQUEST_DELAY_RANGE"]))
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            raise ScrapingError(str(e))
        finally:
            try:
                self.driver.quit()
            except Exception:
                pass

        return self._dedup(all_tweets)
