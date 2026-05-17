# QuantMind — Document de Présentation

> Système hybride **LSTM + Reinforcement Learning** pour la prédiction de prix
> et le trading automatisé de 6 actifs (actions + cryptos), exposé via une API
> FastAPI et un dashboard React.

---

## Sommaire

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture globale](#2-architecture-globale)
3. [Phase 1 — Data Engineering](#3-phase-1--data-engineering)
4. [Phase 2 — Modèle LSTM (Prédiction)](#4-phase-2--modèle-lstm-prédiction)
5. [Phase 3 — Agent RL (Décision)](#5-phase-3--agent-rl-décision)
6. [Phase 4 — Backend FastAPI](#6-phase-4--backend-fastapi)
7. [Phase 5 — Dashboard React](#7-phase-5--dashboard-react)
8. [Phase 6 — DevOps & Livraison](#8-phase-6--devops--livraison)
9. [Résultats expérimentaux](#9-résultats-expérimentaux)
10. [Limites et perspectives](#10-limites-et-perspectives)
11. [Stack technique](#11-stack-technique)
12. [Démo (5 minutes)](#12-démo-5-minutes)

---

## 1. Vue d'ensemble

### 1.1 Problème adressé

Prédire les marchés financiers et automatiser des décisions de trading est un
problème **non-stationnaire**, **bruité** et soumis à des **régimes** (bull /
bear / sideways) qui changent sans prévenir. Les approches purement
statistiques (ARIMA, GARCH) capturent mal les dépendances non-linéaires longues,
et un classifieur supervisé ne gère pas la **dimension séquentielle des
décisions** (acheter aujourd'hui m'empêche d'acheter demain).

### 1.2 Solution proposée

QuantMind combine deux blocs complémentaires :

| Bloc | Rôle | Sortie |
|------|------|--------|
| **LSTM** | Apprend les patterns temporels du prix | Prédiction du Close à J+1 |
| **PPO (RL)** | Apprend la politique de trading optimale | Action ∈ {Hold, Buy, Sell} |

La prédiction LSTM est **injectée comme feature** dans l'observation de l'agent
RL (couplage explicite), ce qui sépare deux compétences :
- *« Que va faire le prix ? »* → LSTM
- *« Que faire de cette information ? »* → RL

### 1.3 Périmètre

- **6 actifs** : AAPL, TSLA, MSFT, GOOGL (actions tech) + BTC-USD, ETH-USD (cryptos)
- **Historique** : 5 ans (≈1 250 jours pour les actions, ≈1 800 jours pour les cryptos)
- **Pipeline complet** : ingestion → indicateurs → entraînement LSTM → entraînement PPO → API → dashboard
- **Dockerisation** : `docker-compose up` lance toute la stack

---

## 2. Architecture globale

```
┌──────────────────────────────────────────────────────────────────────┐
│                      SOURCES DE DONNÉES                              │
│   Kaggle (CSV historique long)        yfinance (compléments + crypto)│
└───────────────┬──────────────────────────────────────┬───────────────┘
                │                                      │
                ▼                                      ▼
        ┌──────────────────────────────────────────────────┐
        │    train.py — step_data()                        │
        │    fusion + dédoublonnage + 5 ans glissants      │
        │    + compute_indicators() (RSI, MACD, EMA, BB…)  │
        └────────────────────┬─────────────────────────────┘
                             │  data/processed/{TICKER}.csv
                             ▼
          ┌──────────────────────────────────────┐
          │   step_lstm()                         │
          │   • MinMaxScaler (fit train uniquement)│
          │   • Séquences glissantes (60 jours)   │
          │   • LSTM 128→64→32 + Dropout/BN       │
          │   • EarlyStopping + ReduceLROnPlateau │
          └────────────────┬─────────────────────┘
                           │  models/lstm_{TICKER}.keras
                           │  models/scaler_{TICKER}.pkl
                           ▼
          ┌──────────────────────────────────────┐
          │   step_rl()                           │
          │   • Génère prédictions LSTM (feature) │
          │   • TradingEnv (gymnasium)            │
          │   • PPO 500 000 timesteps             │
          │   • Backtest vs Buy & Hold            │
          └────────────────┬─────────────────────┘
                           │  models/ppo_{TICKER}.zip
                           │  models/results_{TICKER}.json
                           ▼
          ┌──────────────────────────────────────┐
          │   api.py (FastAPI)                    │
          │   /predict /strategy /performance …   │
          └────────────────┬─────────────────────┘
                           │  HTTP/JSON  (CORS *)
                           ▼
          ┌──────────────────────────────────────┐
          │   frontend/index.html (React + Recharts) │
          │   Dashboard │ Performance │ Données      │
          └──────────────────────────────────────┘
```

---

## 3. Phase 1 — Data Engineering

### 3.1 Acquisition

Deux sources combinées, par ordre de priorité :

1. **Dataset Kaggle « Huge Stock Market Dataset »** (Boris Marjanovic) — historique
   profond (jusqu'à 2017) au format `.txt` mais **encodé en CSV** :
   ```
   Date,Open,High,Low,Close,Volume,OpenInt
   2010-07-21,24.333,24.333,23.946,23.946,43321,0
   ```
   Lu directement avec `pandas.read_csv()` ([train.py:104](train.py#L104)). Le
   format `.txt` ne change rien — c'est un choix d'extension du dataset Kaggle.
2. **yfinance** — complète avec les données récentes (post-2017) et fournit les
   cryptos (BTC, ETH) absentes du dataset Kaggle.

La fusion garde un **historique de 5 ans glissant** par rapport à la date la
plus récente, et déduplique par index temporel (`keep="last"`).

### 3.2 Indicateurs techniques

Calculés dans [train.py:134-188](train.py#L134-L188) :

| Indicateur | Formule | Lecture |
|---|---|---|
| **EMA 20 / 50** | Moyenne exponentielle pondérée | Tendance court / moyen terme |
| **RSI (14)** | `100 − 100 / (1 + gain/loss)` | >70 surachat, <30 survente |
| **MACD (12, 26, 9)** | EMA12 − EMA26, signal = EMA9(MACD) | Croisement haussier/baissier |
| **Bollinger Bands** | Mean(20) ± 2σ + largeur | Mesure de volatilité |
| **ATR (14)** | Moyenne du True Range | Volatilité absolue |
| **Vol_ratio** | Volume / SMA20(Volume) | Anomalies de volume |
| **Returns** | `pct_change`, `log_return` | Rendements journaliers |

Les ~50 premières lignes (warm-up des indicateurs) sont supprimées par
`dropna()`.

### 3.3 EDA — voir [eda.ipynb](eda.ipynb)

Le notebook produit :
- Graphique de performance relative (base 100) des 6 actifs
- Vérification des valeurs manquantes et gaps temporels
- Distribution des returns journaliers (constat : *fat tails*, kurtosis > 3)
- Matrice de corrélation des returns
- Graphique combiné Bollinger / RSI / MACD

### 3.4 Préprocessing LSTM — anti data leakage

Le **MinMaxScaler est fitté uniquement sur les 80 % de train**
([train.py:347-356](train.py#L347-L356)) puis appliqué à l'ensemble. Le split
est **strictement temporel** (pas de shuffle) : 80 % train / 10 % val / 20 %
test.

Les séquences glissantes sont construites avec une fenêtre `SEQ_LEN = 60` :
- Entrée : (60, 10 features)
- Cible : Close à J+61

---

## 4. Phase 2 — Modèle LSTM (Prédiction)

### 4.1 Architecture ([train.py:268-297](train.py#L268-L297))

```
Input(60, 10)
    ↓
LSTM(128, return_sequences=True) + Dropout(0.2)
    ↓
LSTM(64,  return_sequences=True) + Dropout(0.2)
    ↓
LSTM(32,  return_sequences=False) + Dropout(0.2)
    ↓
BatchNormalization()
    ↓
Dense(32, relu)
    ↓
Dense(1)              ← prédiction Close J+1 (espace normalisé)
```

- **Optimiseur** : Adam, lr = 1e-3
- **Loss** : MSE — métrique secondaire MAE
- **Régularisation** : Dropout 0.2 entre chaque LSTM + BatchNorm avant le Dense

### 4.2 Entraînement

Trois callbacks Keras :
- `EarlyStopping(patience=12, restore_best_weights=True)`
- `ModelCheckpoint(save_best_only=True)` → `models/lstm_{TICKER}.keras`
- `ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6)`

Limite : 100 epochs, batch 32. L'EarlyStopping arrête typiquement entre 30 et 60
epochs.

### 4.3 Évaluation

Quatre métriques sur le test set ([train.py:386-394](train.py#L386-L394)) :

| Métrique | Sens |
|---|---|
| **RMSE** | Erreur moyenne en dollars (sensible aux outliers) |
| **MAE** | Erreur absolue moyenne |
| **MAPE** | Erreur en pourcentage du prix réel |
| **Directional Accuracy** | % de fois où la *direction* (↑/↓) est correcte — **métrique clé** |

> **Justification** : prédire le prix exact est très difficile (random walk).
> Prédire la direction est suffisant pour générer une décision de trading.

---

## 5. Phase 3 — Agent RL (Décision)

### 5.1 Environnement Gym ([env.py](env.py))

Implémente l'API `gymnasium.Env` (compatible `stable-baselines3`).

#### Espace d'action — `Discrete(3)`
- `0` = HOLD (rien faire)
- `1` = BUY (acheter le maximum possible, plafonné à `max_shares=10`)
- `2` = SELL (vendre tout)

#### Espace d'observation — `Box(shape=(12,))`
12 features normalisées à chaque pas :

```
[ pred_LSTM/price, RSI/100, MACD/price, EMA20/price, EMA50/price,
  BB_width, ATR/price, Vol_ratio, Return,
  shares/max_shares, balance/initial, portfolio_pnl ]
```

> Le ratio `pred_LSTM/price` est la **clé du couplage** : l'agent voit
> directement *« le LSTM s'attend à un prix X% plus haut/bas demain »*.

#### Reward
```python
reward = (new_value - prev_value) / prev_value
if action == HOLD and shares == 0:
    reward -= 0.0001        # pénalité pour rester immobile
```
Ce `−0.0001` évite la stratégie triviale *« ne jamais trader »*.

#### Frais de transaction réalistes
`TRANSACTION_COST = 0.001` (0.1 %, broker grand public) appliqués à chaque
achat ET vente. Capital initial : 10 000 $.

### 5.2 Validation

```python
from stable_baselines3.common.env_checker import check_env
check_env(TradingEnv(...))   # zéro warning
```

### 5.3 Entraînement PPO ([train.py:549-563](train.py#L549-L563))

Hyperparamètres :
- `learning_rate=3e-4`
- `n_steps=2048`, `batch_size=64`, `n_epochs=10`
- `gamma=0.99`, `gae_lambda=0.95`, `ent_coef=0.01`
- **500 000 timesteps**

EvalCallback toutes les 10 000 steps sur le test set → sauvegarde
`models/best_model.zip` quand un nouveau record est battu.

### 5.4 Backtesting & métriques

Métriques calculées par `TradingEnv.metrics()`
([env.py:112-140](env.py#L112-L140)) :

| Métrique | Définition |
|---|---|
| **Total Return** | `(final / initial − 1) × 100` |
| **Sharpe Ratio** | `mean(returns) / std(returns) × √252` |
| **Max Drawdown** | Plus grosse perte depuis un pic du portefeuille |
| **Win Rate** | % de paires BUY/SELL gagnantes (FIFO) |
| **# Trades** | Nombre total d'ordres |

Comparées à un **Buy & Hold** ([train.py:431-448](train.py#L431-L448)) sur les
mêmes données.

---
 
## 6. Phase 4 — Backend FastAPI

### 6.1 Lifespan ([api.py:25-63](api.py#L25-L63))

Au démarrage, charge en mémoire :
- Tous les `scaler_*.pkl`
- Tous les `lstm_*.keras` (TensorFlow)
- Tous les `ppo_*.zip` (stable-baselines3)
- Tous les `results_*.json`

→ Latence des endpoints **< 200 ms** (hors `/predict` 14 jours qui fait 14
inférences LSTM en boucle).

### 6.2 Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/` | Statut API + modèles chargés |
| GET | `/health` | Healthcheck Docker |
| GET | `/tickers` | 8 tickers supportés |
| GET | `/data/summary?ticker=` | Carte résumée (prix, var, RSI…) |
| GET | `/data?ticker=&period=` | Historique OHLCV + indicateurs |
| GET | `/predict?ticker=&days=N` | Prédictions LSTM auto-régressives J+1…J+N |
| GET | `/strategy?ticker=&period=` | Signaux RL ou règles (fallback) |
| GET | `/performance?ticker=` | Métriques RL vs Buy & Hold |
| GET | `/performance/portfolio?ticker=&period=` | Courbes RL + B&H + distribution returns |

### 6.3 Modes dégradés

Si TensorFlow ou SB3 n'est pas installé, l'API fonctionne quand même :
- `/predict` → `simulate_predictions()` (random walk avec volatilité historique)
- `/strategy` → `rule_based_signals()` (heuristique RSI/MACD/EMA)

→ Le frontend reste utilisable même sur une machine sans GPU.

### 6.4 CORS & Swagger
- `allow_origins=["*"]` (acceptable pour la démo)
- Swagger UI auto-généré : `http://localhost:8000/docs`

---

## 7. Phase 5 — Dashboard React

### 7.1 Stack — *zero-build*

React 18 + Recharts via CDN, Babel standalone pour le JSX. Un seul fichier
[frontend/index.html](frontend/index.html), **aucun npm install requis**. Idéal
pour la démo et la portabilité.

### 7.2 Pages

| Page | Contenu |
|---|---|
| **Guide** | 6 étapes copiables pour démarrer (montrées si API offline) |
| **Dashboard** | 6 cards (prix, var, RSI, signal RL, EMA, compteurs) + graphique combiné prix/EMA/prédiction LSTM + RSI + MACD + tableau des 20 derniers signaux RL |
| **Performance** | Tableau comparatif RL/B&H · cards Total Return · **courbe portfolio RL superposée à B&H** · **histogramme des returns journaliers** |
| **Données** | Table paginée (300 lignes max), filtre par date, **export CSV** |

### 7.3 Sélecteurs globaux
- 6 tickers (sidebar + topbar) regroupés Stock / Crypto
- 6 périodes : 1M, 3M, 6M, 1Y, 2Y, 5Y

### 7.4 État API en temps réel
Polling `/health` toutes les 8 s avec timeout 2.5 s → indicateur LED en bas de
sidebar. Si offline → bannière + redirection sur l'onglet Guide.

---

## 8. Phase 6 — DevOps & Livraison

### 8.1 Stack Docker (2 services)

```yaml
backend:    Python 3.10-slim + uvicorn   →  port 8000
frontend:   nginx 1.27-alpine            →  port 8080
```

### 8.2 Lancement
```bash
docker-compose up --build
```
Une seule commande → API + dashboard accessibles immédiatement.

### 8.3 Healthchecks
- Backend : `curl /health` toutes les 30 s
- Frontend : `wget /` toutes les 30 s
- `start_period: 40s` pour laisser TensorFlow charger les modèles

### 8.4 Volumes en lecture seule
```yaml
volumes:
  - ./models:/app/models:ro
  - ./data:/app/data:ro
```
Sécurité + permet de re-entraîner sur l'host sans reconstruire l'image.

### 8.5 `.dockerignore`
Exclut `data/Stocks/`, `data/ETFs/` (le dataset Kaggle complet pèse plusieurs
Go), `.venv`, `__pycache__`. Image backend finale : ≈ 1.5 Go (TensorFlow CPU
inclus).

---

## 9. Résultats expérimentaux

### 9.1 LSTM — AAPL (test set, 118 jours)

| Métrique | Valeur |
|---|---|
| RMSE | **7.49 $** |
| MAE | **5.82 $** |
| MAPE | **2.18 %** |
| Directional Accuracy | **46.15 %** |
| Epochs (EarlyStop) | 52 |

> *Lecture* : la précision absolue (MAPE 2 %) est correcte mais la
> Directional Accuracy reste proche de 50 %, ce qui rappelle que **prédire le
> sens du marché à J+1 est un problème quasi-aléatoire**. C'est précisément
> pourquoi nous combinons avec un agent RL qui peut gérer le risque même avec
> un signal imparfait.

### 9.2 RL — AAPL (test set)

| Métrique | Agent RL | Buy & Hold |
|---|---|---|
| Total Return (%) | **+9.40** | **+32.59** |
| Sharpe Ratio | **2.15** | 1.36 |
| Max Drawdown (%) | **−1.75** | −13.80 |
| Win Rate (%) | 100 | n/a |
| # Trades | 6 | 1 |

> *Lecture* : l'agent **sous-performe en rendement absolu** (le marché AAPL
> a fortement monté sur la période) mais affiche un **profil risque-ajusté
> nettement supérieur** : Sharpe 1.6× plus élevé, drawdown 8× plus faible,
> 100 % de trades gagnants. C'est exactement le profil recherché en gestion de
> portefeuille institutionnelle.

### 9.3 Caveat scientifique

Ces résultats sont sur **un seul ticker** (AAPL) et **une seule période** de
test (~6 mois fin 2024). Pour une publication, il faudrait :
- Walk-forward validation sur plusieurs splits
- Multi-ticker pour vérifier la généralisation
- Tests statistiques (Diebold-Mariano vs B&H)

---

## 10. Limites et perspectives

### 10.1 Limites assumées

| Limite | Cause | Impact |
|---|---|---|
| **1 modèle par ticker** | Pas de transfer learning | Coût d'entraînement linéaire (×N tickers) |
| **Action discrète {0,1,2}** | Choix de design simple | Pas de sizing fin (toujours all-in / all-out) |
| **Pas de short selling** | Hors scope | Limite la performance en marché baissier |
| **Frais fixes 0.1 %** | Approximation | Réalité : spread + slippage + impact prix |
| **Univers = 8 actifs** | Démo / mémoire | Pas de portefeuille multi-actifs |
| **Pas de stop-loss explicite** | L'agent l'apprend implicitement | Risque de tail event |

### 10.2 Pistes d'amélioration

1. **Transformer** à la place du LSTM (capture des dépendances longues, attention)
2. **Action continue** (`Box[-1, 1]`) → SAC ou TD3 pour le sizing
3. **Multi-asset env** : agent qui alloue entre N actifs simultanément
4. **Reward engineering** : intégrer Sharpe / Sortino dans la reward, pas juste
   la PnL marginale
5. **Régimes de marché** : module de détection (HMM) qui switch entre plusieurs
   politiques RL spécialisées
6. **Données alternatives** : sentiment Twitter, options flow, macro

---

## 11. Stack technique

### 11.1 Backend / ML
| Outil | Version | Rôle |
|---|---|---|
| Python | 3.10 | Runtime |
| TensorFlow | 2.16.1 | LSTM |
| stable-baselines3 | 2.3.0 | PPO |
| gymnasium | 0.29.1 | Environnement RL |
| FastAPI | 0.111.0 | API HTTP |
| uvicorn | 0.29.0 | Serveur ASGI |
| pandas / numpy | 2.2 / 1.26 | Manipulation données |
| scikit-learn | 1.4.2 | MinMaxScaler, métriques |
| yfinance | 0.2.40 | Data provider |

### 11.2 Frontend
| Outil | Rôle |
|---|---|
| React 18 (CDN) | UI |
| Recharts 2.8 | Graphiques |
| Babel standalone | JSX inline |
| nginx 1.27 | Servir le SPA |

### 11.3 DevOps
- Docker 24+ / docker-compose v2
- Multi-stage non utilisé (image backend = single stage Python slim)

---

## 12. Démo (5 minutes)

### Script de démo

1. **Lancement** (10 s)
   ```bash
   docker-compose up --build
   ```
   → Ouvrir `http://localhost:8080`

2. **Page Guide** (30 s)
   - Montrer la check-list avec étapes ✓
   - Indicateur API en bas-gauche → vert

3. **Page Dashboard — AAPL, période 1Y** (90 s)
   - Cards : prix actuel, variation, RSI, dernier signal RL
   - Graphique combiné : prix + EMA20 + ligne pointillée violette = prédiction LSTM
   - Pivoter vers TSLA → tout se recharge dynamiquement
   - Tableau des 20 derniers signaux RL avec couleurs BUY/SELL/HOLD

4. **Page Performance — AAPL** (90 s)
   - Tableau comparatif chiffré RL vs B&H
   - Cards Total Return : RL +9.4 %, B&H +32.6 %
   - **Insister** sur Sharpe (2.15 vs 1.36) et Max DD (−1.75 % vs −13.80 %)
   - Courbe portfolio : RL plus stable, B&H plus volatile
   - Histogramme des returns : *fat tails*, asymétrie

5. **Page Données + export CSV** (30 s)
   - Filtre date `2024-12`
   - Bouton « Export CSV »

6. **Swagger** (30 s)
   - Ouvrir `http://localhost:8000/docs`
   - Tester `/predict?ticker=AAPL&days=14` en direct

7. **Conclusion** (30 s)
   - 3 messages-clés :
     - **Découplage propre** prédiction (LSTM) / décision (RL)
     - **Profil risque-ajusté supérieur** au Buy & Hold
     - **Stack reproductible** en 1 commande Docker

---

## Annexe A — Arborescence du projet

```
quantmind2/
├── train.py                  ← pipeline data + LSTM + RL (3 étapes)
├── env.py                    ← TradingEnv (gymnasium)
├── api.py                    ← FastAPI (9 endpoints)
├── eda.ipynb                 ← notebook exploratoire
├── requirements.txt
├── Dockerfile                ← image backend
├── docker-compose.yml        ← stack 2 services
├── .dockerignore
├── frontend/
│   ├── index.html            ← React + Recharts (zero-build)
│   ├── Dockerfile            ← image nginx
│   └── nginx.conf
├── models/
│   ├── lstm_AAPL.keras       (1.6 MB)
│   ├── scaler_AAPL.pkl
│   ├── ppo_AAPL.zip          (150 KB)
│   └── results_AAPL.json     ← métriques pré-calculées
├── data/
│   ├── Stocks/               ← dataset Kaggle .txt (CSV)
│   ├── ETFs/                 ← dataset Kaggle .txt (CSV, non utilisé)
│   ├── raw/                  ← OHLCV bruts fusionnés
│   └── processed/            ← OHLCV + indicateurs (utilisés par LSTM/RL/API)
├── README.md
└── PRESENTATION.md           ← (ce document)
```

## Annexe B — Format du dataset Kaggle

Le dataset *Huge Stock Market Dataset* utilise l'extension `.txt` mais le
contenu est du **CSV standard** :

```
Date,Open,High,Low,Close,Volume,OpenInt
2010-07-21,24.333,24.333,23.946,23.946,43321,0
2010-07-22,24.644,24.644,24.362,24.487,18031,0
```

Conséquence : `pandas.read_csv()` les lit sans aucune option spéciale. La
colonne `OpenInt` (open interest, pertinente uniquement pour les options) est
ignorée. Couvre `data/Stocks/` (7 195 fichiers) et `data/ETFs/` (1 344
fichiers). QuantMind n'utilise que les actions listées dans `TICKERS_KAGGLE`
([train.py:69-78](train.py#L69-L78)).

## Annexe C — Commandes utiles

```bash
# Pipeline complet pour un ticker (data + LSTM + RL)
python train.py --ticker AAPL --data-path "C:/chemin/vers/Data"

# Étape par étape
python train.py --ticker TSLA --data-path "..." --step data
python train.py --ticker TSLA --step lstm
python train.py --ticker TSLA --step rl

# Tous les tickers d'un coup (long : ~6h)
python train.py --all-tickers --data-path "..."

# API en local (sans Docker)
uvicorn api:app --reload --port 8000

# Stack complète
docker-compose up --build
docker-compose down

# Voir les logs
docker-compose logs -f backend
```

---

*Document généré pour la soutenance — QuantMind v1.0*
