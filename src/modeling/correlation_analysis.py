import pandas as pd
from scipy.stats import pearsonr, spearmanr

from datetime import date

def load_data(sentiment_path, spy_path):
    sentiment_df = pd.read_csv(sentiment_path)
    spy_df = pd.read_csv(spy_path)

    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
    spy_df["Date"] = pd.to_datetime(spy_df["Date"], utc=True).dt.date

    df = pd.merge(sentiment_df, spy_df, left_on="date", right_on="Date", how="inner")
    df["return"] = df["Close"].pct_change()
    df = df.dropna()

    # ⏳ Filtrage entre août et décembre 2024
    start_date = date(2024, 8, 1)
    end_date = date(2024, 12, 31)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    return df

def compute_correlation(df):
    pearson_sentiment_price = pearsonr(df["avg_sentiment"], df["Close"])
    pearson_sentiment_return = pearsonr(df["avg_sentiment"], df["return"])
    spearman_sentiment_price = spearmanr(df["avg_sentiment"], df["Close"])
    spearman_sentiment_return = spearmanr(df["avg_sentiment"], df["return"])

    print("\n📉 Corrélation brute sentiment vs prix SPY :")
    print(f"  Pearson  : {pearson_sentiment_price[0]:.4f}")
    print(f"  Spearman : {spearman_sentiment_price.correlation:.4f}")

    print("\n📈 Corrélation sentiment vs rendement SPY :")
    print(f"  Pearson  : {pearson_sentiment_return[0]:.4f}")
    print(f"  Spearman : {spearman_sentiment_return.correlation:.4f}")

if __name__ == "__main__":
    df = load_data("data/daily_sentiment.csv", "data/spy_prices.csv")
    compute_correlation(df)

