# 📊 Market NLP Strategy Simulator

**Analyse NLP & Marché** — Détection d’anomalies et stratégie ML sur plusieurs actifs financiers.

---

## 🔍 Description

Ce projet combine :

- 🔎 Scraping Reddit pour estimer le **sentiment de marché**
- 🧠 Traitement NLP avec `nltk` et VADER
- ⚠️ Détection d’anomalies via `IsolationForest`
- 📈 Modélisation de stratégie de trading avec `RandomForestClassifier`
- 🎯 Visualisation interactive avec **Streamlit**

Le but : évaluer si le sentiment en ligne améliore une stratégie d’investissement par rapport à un simple buy & hold.

---

## 🗂️ Structure du projet

```bash
.
├── app/                  → Application Streamlit
│   └── streamlit_app.py
├── data/                 → Données sentiment, Reddit, et prix
├── src/                  → Scripts de scraping, prétraitement, modélisation
│   ├── data_collection/
│   ├── modeling/
│   ├── nlp/
│   └── preprocessing/



🚀 Lancer l'application Streamlit

streamlit run app/streamlit_app.py



🔁 Scraping Reddit + Agrégation de sentiment

python src/data_collection/build_asset_datasets.py

📉 Téléchargement des prix via yfinance

python src/data_collection/fetch_price_data.py



Pré-requis

Python 3.10+ streamlit pandas nltk scikit-learn yfinance matplotlib praw

install : pip install -r requirements.txt



Notes
Les fichiers daily_sentiment_<ASSET>_full.csv sont requis pour chaque actif.

L'application gère automatiquement les alertes, le short, et les seuils définis par l’utilisateur.

Si aucun graphique n’apparaît, vérifier que les données sont bien présentes dans data/.





Auteur
Diego Bibal Mazoyer
Étudiant à Centrale Méditerranée
Projet personnel en finance, machine learning & NLP

