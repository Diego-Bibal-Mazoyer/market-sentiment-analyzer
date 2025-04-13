import pandas as pd
from pathlib import Path

# ---------------------- CONFIG ----------------------

# Périodes à inclure
PERIODS_TO_INCLUDE = [
    "aout_dec2024",
    "jan_avr2025"
]

# Dossier des fichiers sentiment par période
DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "daily_sentiment_full.csv"

# -------------------- FUSION ------------------------

def load_and_tag_period(path, period):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = period
    return df

def merge_sentiment_periods():
    all_dfs = []

    for period in PERIODS_TO_INCLUDE:
        file_path = DATA_DIR / f"daily_sentiment_{period}.csv"
        if file_path.exists():
            print(f"✅ Chargement : {file_path.name}")
            df = load_and_tag_period(file_path, period)
            all_dfs.append(df)
        else:
            print(f"⚠️ Fichier manquant : {file_path.name}")

    if not all_dfs:
        print("❌ Aucun fichier valide à fusionner.")
        return

    full_df = pd.concat(all_dfs).sort_values("date")
    full_df = full_df.drop_duplicates(subset="date").reset_index(drop=True)

    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Fusion enregistrée dans : {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_sentiment_periods()

