import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from datetime import date
from pathlib import Path

st.set_page_config(layout="wide")

# ------------------- 📦 Fonctions ------------------- #

def get_available_assets():
    data_path = Path(__file__).parent.parent / "data"
    files = sorted(data_path.glob("daily_sentiment_*_full.csv"))
    assets = sorted(set(
        f.stem.replace("daily_sentiment_", "").replace("_full", "")
        for f in files
    ))
    return assets

def load_data(asset):
    try:
        base_path = Path(__file__).parent.parent / "data"
        sentiment_path = base_path / f"daily_sentiment_{asset}_full.csv"
        price_path = base_path / f"{asset.lower()}_prices.csv"

        sentiment_df = pd.read_csv(sentiment_path)
        price_df = pd.read_csv(price_path, skiprows=[1])  # Ignore la deuxième ligne (celle avec "SPY,SPY,...")


        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
        if "Date" in price_df.columns:
            price_df["Date"] = pd.to_datetime(price_df["Date"], utc=True).dt.date
        else:
            price_df = price_df.reset_index()
            price_df["Date"] = pd.to_datetime(price_df["index"], utc=True).dt.date

        df = pd.merge(sentiment_df, price_df, left_on="date", right_on="Date", how="inner")
        df["return"] = df["Close"].pct_change()
        df["target"] = (df["return"].shift(-1) > 0).astype(int)
        df["sentiment_change"] = df["avg_sentiment"].diff()

        if df.empty or df[["avg_sentiment", "return"]].dropna().empty:
            raise ValueError(f"⚠️ Données vides ou inexploitables pour {asset}")

        model = IsolationForest(contamination=0.1, random_state=42)
        df["anomaly"] = model.fit_predict(df[["avg_sentiment", "return"]].fillna(0))

        return df.dropna().reset_index(drop=True)

    except Exception as e:
        st.warning(str(e))
        return pd.DataFrame()

def compute_alerts(df, threshold):
    return (df["avg_sentiment"] < threshold) | (df["anomaly"] == -1)

def simulate_strategy(df, proba_threshold, use_alerts, enable_short, short_threshold):
    X = df[["avg_sentiment", "sentiment_change", "return", "anomaly"]]
    y = df["target"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    df["proba"] = proba

    if enable_short:
        df["position"] = 0
        df.loc[(proba > proba_threshold) & (df["alert"] == 0), "position"] = 1
        df.loc[df["avg_sentiment"] < short_threshold, "position"] = -1
    elif use_alerts:
        df["position"] = ((proba > proba_threshold) & (df["alert"] == 1)).astype(int)
    else:
        df["position"] = (proba > proba_threshold).astype(int)

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
    exposure = df["position"].abs().sum() / len(df)
    return cum_return, sharpe, drawdown, exposure

# ------------------- 🎛️ Interface ------------------- #

st.title("📊 Analyse NLP & Marché — Multi-actifs, Sentiment & Stratégie ML")

assets = get_available_assets()
if not assets:
    st.error("Aucun fichier de sentiment trouvé. Lancer le scraping d'abord.")
    st.stop()

selected_asset = st.sidebar.selectbox("📈 Actif à analyser", assets)
sentiment_threshold = st.sidebar.slider("Seuil de sentiment (alerte)", -1.0, 1.0, -0.3, 0.05)
proba_threshold = st.sidebar.slider("Seuil proba modèle", 0.4, 0.9, 0.6, 0.05)
use_alerts = st.sidebar.checkbox("🔁 Condition alerte", True)
enable_short = st.sidebar.checkbox("🔻 Activer short si sentiment très négatif", False)
short_threshold = st.sidebar.slider("Seuil sentiment pour short", -1.0, 0.0, -0.7, 0.05)

# ------------------- 📊 Traitement ------------------- #

df = load_data(selected_asset)
if df.empty:
    st.stop()

# Détection des alertes uniquement si nécessaire
# Détection des alertes uniquement si nécessaire
# Par défaut, colonne alert = False
df["alert"] = False

# Cas où alertes sont utilisées (conditionnelles à checkbox ou short)
if use_alerts or enable_short:
    alert_flags = compute_alerts(df, sentiment_threshold)
    if use_alerts:
        df["alert"] = alert_flags  # utilisé pour condition d'entrée
    elif enable_short:
        # On ne met pas à jour df["alert"], mais on envoie `alert_flags` à simulate_strategy
        df["computed_alert"] = alert_flags

df, model = simulate_strategy(df, proba_threshold, use_alerts, enable_short, short_threshold)
ret, sharpe, dd, exposure = compute_metrics(df)

# ------------------- 📈 Visualisation ------------------- #

st.subheader(f"📈 Performance cumulée : {selected_asset}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["date"], df["cum_buy_hold"], label="Buy & Hold", linewidth=2)
ax.plot(df["date"], df["cum_strategy"], label="Stratégie ML", linestyle="--", linewidth=2)
y_min = min(df["cum_buy_hold"].min(), df["cum_strategy"].min()) * 0.98
y_max = max(df["cum_buy_hold"].max(), df["cum_strategy"].max()) * 1.02
ax.set_ylim(y_min, y_max)

# Alertes
ax.scatter(df[df["alert"]]["date"], df[df["alert"]]["cum_buy_hold"], color="red", label="Alertes", marker="x")

# Positions longues
ax.fill_between(df["date"], 0, df["cum_strategy"], where=df["position"]==1, color="green", alpha=0.4, label="Long")

# Positions short
if enable_short:
    ax.fill_between(df["date"], 0, df["cum_strategy"], where=df["position"]==-1, color="red", alpha=0.4, label="Short")

ax.set_ylabel("Performance cumulée")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ------------------- 🧮 KPIs ------------------- #

st.markdown("### 📋 Résumé de performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Return", f"{ret:.2%}")
col2.metric("📏 Sharpe", f"{sharpe:.2f}")
col3.metric("📉 Max Drawdown", f"{dd:.2%}")
col4.metric("⏱️ % Temps Investi", f"{exposure:.2%}")

# ------------------- ℹ️ Note ------------------- #

st.info("⚠️ Les données sentimentales sont spécifiques à chaque actif. Si le graphique est vide, le scraping est sans doute incomplet.")

