import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import os

def load_data(asset, period_key):
    sentiment_path = f"data/daily_sentiment_{asset}_{period_key}.csv"
    price_path = f"data/{asset.lower()}_prices.csv"

    # Chargement des fichiers
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError(f"❌ Fichier sentiment introuvable : {sentiment_path}")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"❌ Fichier prix introuvable : {price_path}")

    sentiment_df = pd.read_csv(sentiment_path)
    price_df = pd.read_csv(price_path)

    # Correction et normalisation de la colonne Date
    if "Date" not in price_df.columns:
        if "Unnamed: 0" in price_df.columns:
            price_df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        elif "index" in price_df.columns:
            price_df.rename(columns={"index": "Date"}, inplace=True)
        else:
            price_df.reset_index(inplace=True)
            if "index" in price_df.columns:
                price_df.rename(columns={"index": "Date"}, inplace=True)

    # Formatage des dates
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
    price_df["Date"] = pd.to_datetime(price_df["Date"], utc=True).dt.date

    # Fusion
    df = pd.merge(sentiment_df, price_df, left_on="date", right_on="Date", how="inner")

    # Check de sécurité
    if df.empty or df[["avg_sentiment", "return"]].dropna().empty:
        raise ValueError(f"⚠️ Données vides ou inexploitables pour {asset} - {period_key}")

    # Calculs
    df["return"] = df["Close"].pct_change()
    df["target"] = (df["return"].shift(-1) > 0).astype(int)
    df["sentiment_change"] = df["avg_sentiment"].diff()

    model = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly"] = model.fit_predict(df[["avg_sentiment", "return"]].fillna(0))

    return df.dropna().reset_index(drop=True)

def compute_alerts(df, sentiment_threshold):
    return (df["avg_sentiment"] < sentiment_threshold) | (df["anomaly"] == -1)

def simulate_strategy(df, proba_threshold, use_alerts, enable_short, short_threshold):
    X = df[["avg_sentiment", "sentiment_change", "return", "anomaly"]]
    y = df["target"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    df["proba"] = proba

    df["position"] = 0  # baseline

    if enable_short:
        # Long : selon proba et option d'alerte
        if use_alerts:
            df.loc[(df["proba"] > proba_threshold) & (~df["alert"]), "position"] = 1
        else:
            df.loc[df["proba"] > proba_threshold, "position"] = 1

        # Short : seuil de sentiment direct
        df.loc[df["avg_sentiment"] < short_threshold, "position"] = -1

    else:
        if use_alerts:
            df["position"] = ((df["proba"] > proba_threshold) & df["alert"]).astype(int)
        else:
            df["position"] = (df["proba"] > proba_threshold).astype(int)

    df["strategy_return"] = df["return"] * df["position"]
    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_buy_hold"] = (1 + df["return"]).cumprod()
    return df, model



def compute_metrics(df):
    strat = df["strategy_return"]
    cum_return = df["cum_strategy"].iloc[-1] - 1
    volatility = strat.std() * np.sqrt(252)
    sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
    drawdown = (df["cum_strategy"].cummax() - df["cum_strategy"]).max()
    invested_pct = df["position"].sum() / len(df)
    return cum_return, sharpe, drawdown, invested_pct

