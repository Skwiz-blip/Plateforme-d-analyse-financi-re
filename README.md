# QuantMind -- Guide Complet

QuantMind est un **système intelligent qui apprend à prédire les prix des actions et à prendre des décisions d'achat/vente automatiques**.

Il combine deux technologies :
- **LSTM** : Un réseau de neurones qui prédit le prix futur
- **RL (PPO)** : Un agent qui apprend quand acheter ou vendre

---

## Comment ça fonctionne ? (Explication simple)

### Étape 1 : Récupération des données

```
Données historiques (prix, volume) 
    --> Indicateurs techniques calculés (RSI, MACD, EMA...)
    --> Sauvegardées dans data/processed/
```

**En simple** : On télécharge l'historique des prix (5 ans) et on calcule des "indicateurs" qui aident à comprendre si le marché monte ou descend.

### Étape 2 : Entraînement du LSTM (Prédiction)

```
60 jours d'historique --> LSTM --> Prédiction du prix J+1
```

**En simple** : Le LSTM regarde les 60 derniers jours et essaie de deviner le prix de demain. Il s'entraîne sur des milliers d'exemples jusqu'à devenir bon.

**Métriques importantes** :
- `Directional Accuracy` : % de fois où il devine la bonne direction (hausse/baisse)

### Étape 3 : Entraînement de l'agent RL (Décision)

```
État du marché + Prédiction LSTM --> Agent RL --> Action (Acheter/Vendre/Garder)
```

**En simple** : L'agent apprend à trader en simulant des milliers de journées de bourse. Il gagne des "récompenses" quand il fait des bénéfices.

**Important** : 
- L'agent utilise la prédiction du LSTM comme information
- Il y a des **frais de transaction (0.1%)** pour être réaliste
- Il est entraîné sur **500 000 simulations**

---

## Structure des fichiers

```
quantmind/
+-- train.py          <- Script d'entraînement (3 étapes)
+-- env.py            <- Environnement de trading (gym)
+-- api.py            <- API pour utiliser les modèles
+-- frontend/
|   +-- index.html    <- Interface web
+-- models/           <- Modèles sauvegardés
|   +-- lstm_AAPL.keras    <- Réseau LSTM entraîné
|   +-- scaler_AAPL.pkl    <- Normalisation
|   +-- ppo_AAPL.zip       <- Agent RL entraîné
|   +-- results_AAPL.json  <- Métriques de performance
+-- data/
|   +-- raw/          <- Données brutes
|   +-- processed/    <- Données avec indicateurs
+-- requirements.txt  <- Dépendances Python
```

---

## Installation et Utilisation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Télécharger les données

**Option A** : Dataset Kaggle (recommandé)
- Télécharger : https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
- Placer dans un dossier `Data/`

**Option B** : yfinance automatique
- Le script télécharge automatiquement si pas de fichier local

### 3. Lancer l'entraînement

```bash
# Pipeline complet (recommandé)
python train.py --ticker AAPL --data-path "C:/chemin/vers/Data"

# Ou étape par étape
python train.py --ticker AAPL --data-path "C:/chemin/vers/Data" --step data
python train.py --ticker AAPL --step lstm
python train.py --ticker AAPL --step rl
```

### 4. Lancer l'API

```bash
uvicorn api:app --reload --port 8000
```

### 5. Utiliser l'interface web

Ouvrir `frontend/index.html` dans un navigateur.

---

## Endpoints de l'API

| Endpoint | Description |
|----------|-------------|
| `GET /` | État de l'API |
| `GET /tickers` | Liste des actions disponibles |
| `GET /data?ticker=AAPL` | Données historiques |
| `GET /predict?ticker=AAPL&days=14` | Prédictions LSTM |
| `GET /strategy?ticker=AAPL` | Signaux d'achat/vente |
| `GET /performance?ticker=AAPL` | Performance vs Buy & Hold |

---

## Indicateurs Techniques Utilisés

| Indicateur | Signification simple |
|------------|---------------------|
| **RSI** | Entre 0-100. >70 = suracheté, <30 = survendu |
| **MACD** | Différence entre 2 moyennes mobiles. >0 = tendance haussière |
| **EMA_20/50** | Moyenne mobile exponentielle. Prix au-dessus = haussier |
| **BB_width** | Largeur des bandes de Bollinger. Élevé = forte volatilité |
| **ATR** | Mesure la volatilité moyenne |
| **Vol_ratio** | Volume vs moyenne. >1 = volume élevé |

---

## Métriques de Performance

### Pour le LSTM
- **RMSE** : Erreur moyenne en dollars
- **MAPE** : Erreur moyenne en pourcentage
- **Directional Accuracy** : % de bonnes prédictions de direction

### Pour l'agent RL
- **Total Return** : Gain/perte total en %
- **Sharpe Ratio** : Rendement ajusté au risque (>1 = bon)
- **Max Drawdown** : Plus grosse perte depuis un pic
- **Win Rate** : % de trades gagnants

---

## Prévention du Data Leakage

**Problème courant** : Utiliser des données futures pour prédire le passé.

**Notre solution** :
1. Split temporel strict (80% train, 20% test)
2. Le scaler est fitté uniquement sur le train set
3. Pas de shuffle des données

---

## Architecture du Système

```
                    +------------------+
                    |   Données CSV    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Indicateurs Tech |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
    +------------------+           +------------------+
    |   LSTM Model     |           |   Agent RL       |
    | (Prédiction)     |---------->| (Décision)       |
    +------------------+  feature  +------------------+
                                    |
                                    v
                           +------------------+
                           |   Action:        |
                           | 0=Hold 1=Buy     |
                           | 2=Sell           |
                           +------------------+
```

---

## Tickers Supportés

| Ticker | Nom | Type |
|--------|-----|------|
| AAPL | Apple | Action |
| TSLA | Tesla | Action |
| MSFT | Microsoft | Action |
| GOOGL | Alphabet | Action |
| AMZN | Amazon | Action |
| NVDA | NVIDIA | Action |
| BTC-USD | Bitcoin | Crypto |
| ETH-USD | Ethereum | Crypto |

---

## Dépannage

### Erreur : "gymnasium non disponible"
```bash
pip install gymnasium
```

### Erreur : "stable-baselines3 non disponible"
```bash
pip install stable-baselines3
```

### Erreur : "TensorFlow non disponible"
```bash
pip install tensorflow
```

### L'agent trade trop souvent
C'est normal si les frais de transaction n'étaient pas activés. Ils le sont maintenant (0.1%).

---

## Pour aller plus loin

- Augmenter `total_timesteps` dans `train.py` pour un meilleur entraînement RL
- Ajouter d'autres indicateurs techniques dans `FEATURES`
- Modifier `TRANSACTION_COST` dans `env.py` pour simuler d'autres frais
- Ajuster les hyperparamètres PPO (learning_rate, batch_size...)
