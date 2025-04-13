import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

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

def detect_anomalies(df):
    model = IsolationForest(contamination=0.1, random_state=42)

    features = df[["avg_sentiment", "return"]].fillna(0)
    df["anomaly"] = model.fit_predict(features)

    return df

def plot_anomalies(df):
    fig, ax = plt.subplots(figsize=(12,6))

    normal = df[df["anomaly"] == 1]
    outliers = df[df["anomaly"] == -1]

    ax.plot(df["date"], df["Close"], label="SPY Close", color="blue")
    ax.scatter(outliers["date"], outliers["Close"], color="red", label="Anomalies", zorder=5)
    ax.set_title("Détection d'anomalies SPY vs Sentiment")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix SPY")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data("data/daily_sentiment.csv", "data/spy_prices.csv")
    df = detect_anomalies(df)
    plot_anomalies(df)

