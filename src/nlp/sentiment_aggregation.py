import pandas as pd

def aggregate_sentiment_by_day(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Convertir les dates en datetime
    df["created_utc"] = pd.to_datetime(df["created_utc"])

    # Grouper par jour
    df["date"] = df["created_utc"].dt.date
    daily_sentiment = df.groupby("date")["sentiment"].mean().reset_index()
    daily_sentiment.columns = ["date", "avg_sentiment"]

    daily_sentiment.to_csv(output_path, index=False)
    print(f"{len(daily_sentiment)} jours de sentiment sauvegardés → {output_path}")

if __name__ == "__main__":
    aggregate_sentiment_by_day("data/reddit_sp500_sentiment.csv", "data/daily_sentiment.csv")

