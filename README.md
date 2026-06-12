# QuantMind — Plateforme d'analyse financière (stocks)

**Sujet 20 — Projets AI & Big Data** : système complet de bout en bout
`Données → Ingestion (Kafka) → Modèles (LSTM + RL) → API (FastAPI) → Application web → Docker`

QuantMind apprend à **prédire les rendements boursiers** et à **prendre des décisions
d'achat/vente automatiques**, puis rejoue le marché **en temps réel** via un pipeline de streaming.

Il combine :
- **LSTM** : prédit le *rendement* du lendemain (hausse/baisse et amplitude)
- **RL (PPO)** : un agent qui apprend quand acheter, vendre ou attendre
- **Kafka** : flux de prix temps réel → scoring live → dashboard

---

## 🚀 Démarrage en UNE commande (recommandé)

```bash
docker compose up --build
```

Cela démarre les 5 services :

| Service | Rôle | Accès |
|---|---|---|
| `kafka` | Bus de messages (ingestion temps réel) | interne (`:9094` pour le dev local) |
| `producer` | Rejoue l'historique des prix tick par tick → topic `prices` | — |
| `consumer` | Score chaque tick (LSTM + PPO) + paper trading → topic `signals` | — |
| `backend` | API FastAPI (modèles + endpoints live) | http://localhost:8000 (docs : `/docs`) |
| `frontend` | Dashboard web (React + Recharts) | http://localhost:8080 |

Ouvre **http://localhost:8080** → onglet **Live** pour voir les signaux tomber en temps réel.

---

## Architecture du système

```
              ┌─────────────── Conteneurs Docker ────────────────────────┐
              │                                                          │
 Données      │  ┌──────────┐   ┌─────────┐   ┌──────────────────┐      │
 Kaggle ────────►│ Producer │──►│  Kafka  │──►│     Consumer     │      │
 (CSV)        │  │ (replay) │   │ prices  │   │ LSTM + PPO live  │      │
              │  └──────────┘   │ signals │◄──│ + paper trading  │      │
              │                 └────┬────┘   └──────────────────┘      │
              │                      │                                  │
              │                 ┌────▼────────┐      ┌──────────┐       │
              │                 │ API FastAPI │─────►│ Frontend │───────┼──► Utilisateur
              │                 │ /predict    │      │ React    │       │
              │   models/ ─────►│ /strategy   │      │ + Live   │       │
              │   (entraînés)   │ /live/*     │      └──────────┘       │
              │                 └─────────────┘                         │
              └──────────────────────────────────────────────────────────┘
```

---

## Comment ça fonctionne

### Étape 1 — Données + indicateurs techniques
On télécharge l'historique des prix (dataset Kaggle) et on calcule les indicateurs :
RSI, MACD, EMA 20/50, Bandes de Bollinger, ATR, ratio de volume.

### Étape 2 — LSTM : prédire le RENDEMENT, pas le prix
Le LSTM regarde les **60 derniers jours** et prédit le **rendement de demain (%)**.

> **Pourquoi le rendement et pas le prix ?** Le prix est non-stationnaire : un modèle
> entraîné sur des prix à 100$ dérive quand le titre passe à 170$. Le rendement
> journalier (~±1%) est stationnaire → le modèle apprend la *variation*, qui est la
> vraie inconnue. Le prix est ensuite reconstruit : `prix_prédit = Close_J × (1 + r̂/100)`.

Le modèle est **toujours comparé à des baselines** sur le même test set :

| Modèle (AAPL, test set) | RMSE ($) | MAPE % | Dir. Acc % |
|---|---|---|---|
| Naïve (persistance, r̂=0) | 1.88 | 0.90 | — |
| Momentum (r̂ = rendement J-1) | 2.53 | 1.25 | 47.9 |
| Régression linéaire | 1.94 | 0.95 | 42.9 |
| **LSTM (rendement)** | **1.91** | **0.90** | **56.3** |

→ Le LSTM égale la naïve sur le prix et **bat nettement le hasard sur la direction**,
qui est l'information réellement exploitée par l'agent RL.

### Étape 3 — Agent RL (PPO) : décider
```
État du marché + rendement prédit (LSTM) → Agent PPO → BUY / SELL / HOLD
```
- Entraîné sur 500 000 simulations (environnement Gym custom)
- **Frais de transaction 0.1%** pour rester réaliste
- Évalué vs **Buy & Hold** (return total, Sharpe, max drawdown, win rate)

### Étape 4 — Streaming temps réel (Kafka)
- `streaming/producer.py` rejoue l'historique tick par tick (topic `prices`)
- `streaming/consumer.py` maintient une fenêtre glissante de 60 jours par ticker,
  score chaque tick (LSTM + PPO avec **probabilité réelle** de la policy) et gère
  un **portefeuille virtuel** (paper trading) → topic `signals`
- L'API expose `/live/status` et `/live/signals` → onglet **Live** du dashboard

---

## Structure des fichiers

```
quantmind/
├── train.py              ← Pipeline d'entraînement (data → LSTM → RL)
├── env.py                ← Environnement de trading (Gym)
├── api.py                ← API FastAPI (+ endpoints live)
├── streaming/
│   ├── producer.py       ← Replay des prix → Kafka topic `prices`
│   └── consumer.py       ← Scoring temps réel → Kafka topic `signals`
├── frontend/
│   └── index.html        ← Dashboard web (onglets Dashboard / Live / Modèle / ...)
├── models/               ← Modèles entraînés (lstm_*.keras, scaler_*.pkl, ppo_*.zip)
├── data/processed/       ← Données + indicateurs techniques
├── Dockerfile            ← Image backend / producer / consumer
├── docker-compose.yml    ← Orchestration des 5 services
└── requirements.txt
```

---

## Entraînement des modèles

### 1. Dataset Kaggle
Télécharger : https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
(le dossier `Data/` contient `Stocks/` et `ETFs/`)

### 2. Pipeline complet

```bash
pip install -r requirements.txt

# Un ticker (données → LSTM → RL)
python train.py --ticker AAPL --data-path "C:/chemin/vers/Data"

# Tous les tickers
python train.py --all-tickers --data-path "C:/chemin/vers/Data"

# Ou étape par étape
python train.py --ticker AAPL --step data --data-path "C:/chemin/vers/Data"
python train.py --ticker AAPL --step lstm
python train.py --ticker AAPL --step rl
```

> ⚠️ **Important** : depuis le passage à la cible *rendement*, les anciens modèles
> (qui prédisaient le prix brut) sont incompatibles. Réentraîner LSTM **puis** RL
> pour chaque ticker (le RL consomme les prédictions du LSTM).

### 3. Lancement manuel (sans Docker)

```bash
uvicorn api:app --port 8000                     # API
# Mode live (optionnel) — nécessite Kafka :
docker compose up -d kafka
$env:KAFKA_BOOTSTRAP="localhost:9094"; python streaming/consumer.py
$env:KAFKA_BOOTSTRAP="localhost:9094"; python streaming/producer.py
$env:KAFKA_BOOTSTRAP="localhost:9094"; uvicorn api:app --port 8000
```

---

## Endpoints de l'API

| Endpoint | Description |
|---|---|
| `GET /` , `GET /health` | État de l'API |
| `GET /tickers` | Tickers avec modèles entraînés |
| `GET /data?ticker=AAPL&period=1Y` | Données historiques + indicateurs |
| `GET /data/summary?ticker=AAPL` | Résumé (prix, RSI, MACD...) |
| `GET /predict?ticker=AAPL&days=14` | Prévision LSTM (rendements → prix reconstruits) |
| `GET /strategy?ticker=AAPL` | Signaux BUY/SELL/HOLD + **probabilité réelle** PPO |
| `GET /performance?ticker=AAPL` | Backtest RL vs Buy & Hold |
| `GET /model/info?ticker=AAPL` | Architecture LSTM + hyperparamètres PPO + métriques |
| `GET /live/status` | État du pipeline streaming Kafka |
| `GET /live/signals?ticker=AAPL` | Derniers signaux temps réel (mode Live) |

---

## Méthodologie (anti data-leakage)

1. **Split temporel strict** 80/10/10 (train/val/test) — jamais de shuffle
2. Scaler **fitté sur le train uniquement**
3. Le test set n'est utilisé **qu'une seule fois**, à la fin
4. Baselines évaluées sur le **même** test set que le LSTM
5. Backtest RL sur des données jamais vues, frais de transaction inclus

---

## Limites connues (voir rapport, section 8)

- La prévision multi-jours (`/predict?days=N`) gèle les indicateurs techniques
  (seul le prix est mis à jour de façon autorégressive)
- Le dataset Kaggle s'arrête en 2017 (le mode Live **rejoue** ce flux historique)
- Pas d'analyse fondamentale (P/E, bilans) — données historiques non disponibles gratuitement
- Améliorations possibles : ingestion yfinance (données récentes + crypto),
  features de contexte marché (S&P 500, VIX), job Spark batch sur les ~8 000 tickers

---

## Dépannage

| Problème | Solution |
|---|---|
| `gymnasium` / `stable-baselines3` / `tensorflow` manquant | `pip install -r requirements.txt` |
| `docker pull` / `docker compose build` échoue (DNS `auth.docker.io`) | récupérer chaque image via le miroir Google puis la retaguer : `docker pull mirror.gcr.io/<image>` puis `docker tag mirror.gcr.io/<image> <image>`. Images nécessaires : `apache/kafka:3.7.2`, `library/python:3.10-slim`, `library/nginx:1.27-alpine` |
| Onglet Live : "Kafka injoignable" | le broker met ~20s à démarrer, retry automatique |
| `/predict` renvoie 503 | modèles non entraînés pour ce ticker → `python train.py --ticker X` |
