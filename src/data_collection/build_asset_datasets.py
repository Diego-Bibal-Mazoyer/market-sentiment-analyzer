import os
import praw
import pandas as pd
import re
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

# --------------- CONFIG PRAW --------------- #

REDDIT_CLIENT_ID = "t0335GOESRGGPMtNS4QFEQ"
REDDIT_SECRET = "Adjb9_rqFhYDkTXGeH7KUHgB4HMQbw"
REDDIT_USER_AGENT = "market-detector by /u/Diego-Bibal-Mazoyer"

POST_LIMIT = 1000

# --------------- ACTIFS À SCRAPER --------------- #

ASSETS = {
    "SPY": {
        "subreddits": ["wallstreetbets", "investing", "stocks", "SPY"],
        "keywords": ["s&p", "sp500", "spy", "index"]
    },
    "BTC": {
        "subreddits": ["cryptocurrency", "Bitcoin", "CryptoMarkets"],
        "keywords": ["btc", "bitcoin", "satoshi", "crypto"]
    },
    "QQQ": {
        "subreddits": ["wallstreetbets", "investing", "stocks"],
        "keywords": ["qqq", "nasdaq", "tech index"]
    },
    "TSLA": {
        "subreddits": ["wallstreetbets", "TeslaMotors", "stocks"],
        "keywords": ["tsla", "tesla", "elon", "model 3"]
    }
}

# --------------- PÉRIODES À SCRAPER --------------- #

PERIODS = {
    "aout_dec2024": ("2024-08-01", "2024-12-31"),
    "janv_avr2025": ("2025-01-01", "2025-04-11"),
    "mars2025": ("2025-03-01", "2025-03-31")
}

# --------------- NLP & UTILS --------------- #

sia = SentimentIntensityAnalyzer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentiment_score(text):
    if not text:
        return 0.0
    return sia.polarity_scores(text)["compound"]

# --------------- SCRAPING --------------- #

def fetch_posts(subreddit_name, keywords, after, before, limit=1000):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.new(limit=limit * 5):
        created = submission.created_utc
        if after <= created <= before:
            full_text = f"{submission.title} {submission.selftext}"
            if any(k in full_text.lower() for k in keywords):
                posts.append({
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": datetime.fromtimestamp(submission.created_utc),
                    "subreddit": subreddit_name
                })
        if len(posts) >= limit:
            break
    return posts

def build_asset_period_dataset(asset, config, period_name, after_ts, before_ts):
    all_posts = []
    print(f"\n🔍 Scraping {asset} pour {period_name}...")

    for sub in config["subreddits"]:
        posts = fetch_posts(sub, config["keywords"], after_ts, before_ts, POST_LIMIT)
        all_posts.extend(posts)

    df = pd.DataFrame(all_posts)
    if df.empty:
        print(f"⚠️ Aucun post trouvé pour {asset} — {period_name}")
        return

    df["full_text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
    df["clean_text"] = df["full_text"].apply(clean_text)
    df["sentiment"] = df["clean_text"].apply(sentiment_score)
    df["date"] = pd.to_datetime(df["created_utc"]).dt.date

    df.to_csv(f"data/reddit_{asset}_{period_name}.csv", index=False)

    agg = df.groupby("date")["sentiment"].mean().reset_index()
    agg.columns = ["date", "avg_sentiment"]
    agg.to_csv(f"data/daily_sentiment_{asset}_{period_name}.csv", index=False)

    print(f"✅ Fichier enregistré : data/daily_sentiment_{asset}_{period_name}.csv ({len(agg)} jours)")

# --------------- MAIN --------------- #

if __name__ == "__main__":
    for asset, config in ASSETS.items():
        for period_name, (start, end) in PERIODS.items():
            after_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
            before_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp())
            build_asset_period_dataset(asset, config, period_name, after_ts, before_ts)

