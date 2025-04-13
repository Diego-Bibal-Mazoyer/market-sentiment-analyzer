import yfinance as yf
import pandas as pd

ASSETS = {
    "SPY": "SPY",
    "BTC": "BTC-USD",
    "QQQ": "QQQ",
    "TSLA": "TSLA"
}

START = "2024-08-01"
END = "2025-04-11"

for symbol, ticker in ASSETS.items():
    print(f"📥 Téléchargement : {symbol}")
    data = yf.download(ticker, start=START, end=END)

    # Vérifie que ce n'est pas vide
    if data.empty:
        print(f"⚠️ Pas de données pour {symbol}")
        continue

    # Assure-toi que l'index est une colonne normale
    data.reset_index(inplace=True)

    # Sauvegarde sans multi-index, sans colonne 'Ticker'
    data.to_csv(f"data/{symbol.lower()}_prices.csv", index=False)
    print(f"✅ Sauvegardé : data/{symbol.lower()}_prices.csv")

