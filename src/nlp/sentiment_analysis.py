import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def apply_vader(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return sia.polarity_scores(text)["compound"]

def run_sentiment_analysis(input_path, output_path):
    df = pd.read_csv(input_path)
    df["sentiment"] = df["clean_text"].apply(apply_vader)
    df.to_csv(output_path, index=False)
    print(f"Sentiment ajouté à {len(df)} lignes → sauvegardé dans {output_path}")

if __name__ == "__main__":
    run_sentiment_analysis("data/reddit_sp500_clean.csv", "data/reddit_sp500_sentiment.csv")

