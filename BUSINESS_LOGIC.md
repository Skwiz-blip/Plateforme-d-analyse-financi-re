# QuantMind -- Logique Métier Complète

## Vue d'ensemble du Système de Trading

QuantMind est un **système de trading algorithmique hybride** qui combine deux approches complémentaires :

1. **Prédiction de prix (LSTM)** : Anticiper la direction du marché
2. **Prise de décision (RL)** : Déterminer le moment optimal d'achat/vente

---

## 1. Logique de Prédiction -- LSTM

### Principe de Fonctionnement

Le LSTM (Long Short-Term Memory) est un réseau de neurones spécialisé dans l'analyse de séquences temporelles.

```
Entrée : 60 jours d'historique (prix, volume, indicateurs)
Sortie : Prix prédit pour J+1
```

### Pourquoi 60 jours ?

- **Court terme** : Capture les tendances immédiates
- **Moyen terme** : Intègre les cycles de marché
- **Historique suffisant** : Pour détecter les patterns récurrents

### Features Utilisées (10 variables)

| Feature | Rôle dans la Prédiction |
|---------|------------------------|
| **Close** | Prix de clôture -- cible principale |
| **Volume** | Intérêt du marché, validation des mouvements |
| **EMA_20** | Tendance court terme lissée |
| **EMA_50** | Tendance moyen terme lissée |
| **RSI** | Niveau de surachat/survente |
| **MACD** | Momentum et changement de tendance |
| **MACD_signal** | Confirmation du signal MACD |
| **BB_width** | Volatilité actuelle du marché |
| **ATR** | Amplitude des mouvements quotidiens |
| **Vol_ratio** | Volume anormal vs habituel |

### Processus de Normalisation

```
Données brutes --> MinMaxScaler (0-1) --> LSTM --> Dé-normalisation --> Prix réel
```

**Important** : Le scaler est fitté UNIQUEMENT sur les données d'entraînement (80%) pour éviter le data leakage.

### Métriques de Qualité

| Métrique | Interprétation Financière |
|----------|--------------------------|
| **RMSE** | Erreur moyenne en dollars -- à minimiser |
| **MAE** | Erreur absolue moyenne -- plus stable |
| **MAPE** | Erreur en % -- comparable entre actions |
| **Directional Accuracy** | % de bonnes directions -- LE PLUS IMPORTANT |

**Pourquoi Directional Accuracy est crucial ?**

En trading, connaître la direction (hausse/baisse) est plus important que le prix exact. Un modèle avec 55%+ de directional accuracy peut être profitable.

---

## 2. Logique de Décision -- Agent RL (PPO)

### Principe du Reinforcement Learning

L'agent apprend par **essai-erreur** dans un environnement simulé :

```
État --> Action --> Récompense --> Nouvel État --> ...
```

Il optimise ses actions pour maximiser les récompenses cumulées.

### Espace d'Actions (3 actions)

| Action | Code | Condition |
|--------|------|-----------|
| **HOLD** | 0 | Ne rien faire |
| **BUY** | 1 | Si balance >= prix ET shares < max |
| **SELL** | 2 | Si shares > 0 |

### Espace d'Observation (12 variables)

L'agent "voit" le marché à travers 12 informations :

```
Observation = [
    prédiction_LSTM / prix_actuel,     # Couplage LSTM-RL
    RSI / 100,                         # Normalisé 0-1
    MACD / prix,                       # Normalisé
    EMA_20 / prix,                     # Ratio
    EMA_50 / prix,                     # Ratio
    BB_width,                          # Volatilité
    ATR / prix,                        # Volatilité normalisée
    Vol_ratio,                         # Volume anormal
    Return,                            # Rendement précédent
    shares / max_shares,               # Position actuelle
    balance / initial_balance,         # Capital restant
    performance_cumulée - 1,           # P&L relatif
]
```

### Fonction de Récompense

```python
reward = variation_portfolio / valeur_précédente

# Pénalité pour inactivité sans position
if action == HOLD and shares == 0:
    reward -= 0.0001
```

**Logique** :
- Récompense positive si le portfolio augmente
- Pénalité légère pour inactivité (évite de ne jamais trader)

### Frais de Transaction (0.1%)

```python
TRANSACTION_COST = 0.001  # 0.1%

# Achat
coût = quantité × prix × 0.001
balance -= quantité × prix + coût

# Vente
produit = shares × prix
coût = produit × 0.001
balance += produit - coût
```

**Impact** :
- Décourage le trading excessif
- Simule la réalité du marché
- L'agent apprend à ne trader que si le gain > frais

### Algorithme PPO (Proximal Policy Optimization)

PPO est un algorithme de policy gradient qui :
- Met à jour la politique de manière conservative
- Évite les changements trop brutaux
- Est stable et efficace pour le trading

**Hyperparamètres** :
- `learning_rate = 3e-4` : Vitesse d'apprentissage
- `n_steps = 2048` : Pas avant mise à jour
- `batch_size = 64` : Taille du batch
- `gamma = 0.99` : Importance du futur
- `gae_lambda = 0.95` : Équilibrage advantage
- `ent_coef = 0.01` : Encouragement à l'exploration

---

## 3. Couplage LSTM --> RL

### Architecture Hybride

```
                    +-----------------+
                    |   Marché (OHLCV)|
                    +--------+--------+
                             |
                             v
                    +-----------------+
                    |  Indicateurs    |
                    |  Techniques     |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
    +-----------------+            +-----------------+
    |    LSTM         |            |   État RL       |
    | (Prédiction)    |----------->| (Observation)   |
    +-----------------+  feature   +-----------------+
                                    |
                                    v
                           +-----------------+
                           |  Agent PPO      |
                           |  (Décision)     |
                           +-----------------+
                                    |
                                    v
                           +-----------------+
                           |  Action:        |
                           | BUY/SELL/HOLD   |
                           +-----------------+
```

### Pourquoi ce Couplage ?

1. **LSTM** : Fournit une "opinion" sur la direction future
2. **RL** : Intègre cette opinion avec d'autres signaux pour décider

L'agent RL ne se base pas uniquement sur la prédiction LSTM, il l'utilise comme **une feature parmi d'autres**.

### Intégration dans l'Observation

```python
# La prédiction LSTM normalisée par le prix actuel
obs[0] = lstm_prediction / current_price

# Si prédiction > prix --> signal d'achat potentiel
# Si prédiction < prix --> signal de vente potentiel
```

---

## 4. Gestion du Risque

### Contraintes Intégrées

| Contrainte | Valeur | Objectif |
|------------|--------|----------|
| **max_shares** | 10 | Limiter l'exposition |
| **initial_balance** | $10,000 | Capital de départ |
| **transaction_cost** | 0.1% | Réalisme |

### Métriques de Risque

**Sharpe Ratio**
```
Sharpe = (rendement_moyen / écart_type) × sqrt(252)
```
- > 1 : Bon
- > 2 : Très bon
- > 3 : Excellent

**Max Drawdown**
```
MaxDD = max(perte depuis le plus haut historique)
```
- Mesure le risque de ruine
- À minimiser absolument

### Split Temporel (Anti-Overfitting)

```
|---- Train (80%) ----|---- Test (20%) ----|
     Entraînement         Évaluation
```

- Pas de shuffle
- Test sur données futures (jamais vues)
- Validation honnête de la performance

---

## 5. Cycle de Vie d'un Trade

### Exemple Complet

```
Jour 0 : 
  - État : Balance $10,000, 0 shares
  - Observation : RSI=30, LSTM prédit hausse
  - Action : BUY 5 shares @ $150
  - Coût : $750 + $0.75 (frais) = $750.75
  - Nouvel état : Balance $9,249.25, 5 shares

Jour 1 :
  - Prix monte à $155
  - Portfolio = $9,249.25 + (5 × $155) = $10,024.25
  - Reward = ($10,024.25 - $10,000) / $10,000 = +0.24%

Jour 5 :
  - Prix = $160
  - Action : SELL 5 shares @ $160
  - Produit : $800 - $0.80 (frais) = $799.20
  - Nouvel état : Balance $10,048.45, 0 shares
  - Gain total : +0.48%
```

### Calcul du Win Rate

```python
# Paires BUY --> SELL dans l'ordre chronologique
pairs = []
buy_queue = []

for trade in trades:
    if trade.type == "BUY":
        buy_queue.append(trade)
    elif trade.type == "SELL" and buy_queue:
        buy = buy_queue.pop(0)  # FIFO
        pairs.append((buy, sell))

wins = sum(1 for buy, sell in pairs if sell.price > buy.price)
win_rate = wins / len(pairs) × 100
```

---

## 6. Comparaison vs Buy & Hold

### Stratégie Buy & Hold

```
Acheter au début, vendre à la fin
Performance = (prix_final / prix_initial - 1) × 100
```

### Pourquoi Comparer ?

- **Benchmark standard** du marché
- Si l'agent RL bat le Buy & Hold --> valeur ajoutée
- Si l'agent RL perd --> mieux vaut ne rien faire

### Métriques de Comparaison

| Métrique | Agent RL | Buy & Hold | Interprétation |
|----------|----------|------------|----------------|
| Total Return | X% | Y% | Performance brute |
| Sharpe Ratio | X | Y | Rendement ajusté au risque |
| Max Drawdown | X% | Y% | Risque maximum |
| Nb Trades | N | 1 | Activité |

---

## 7. Flux de Données Complet

```
1. CHARGEMENT
   Kaggle/yfinance --> CSV (OHLCV)

2. INDICATEURS
   CSV --> compute_indicators() --> CSV + features

3. NORMALISATION
   Features --> MinMaxScaler --> [0, 1]

4. SÉQUENCES
   Data --> sliding window (60) --> X, y

5. LSTM TRAINING
   X_train, y_train --> model.fit() --> lstm.keras

6. PREDICTIONS
   Data --> lstm.predict() --> prédictions

7. RL ENVIRONMENT
   Data + prédictions --> TradingEnv

8. RL TRAINING
   TradingEnv --> PPO.learn() --> ppo.zip

9. INFÉRENCE
   Nouvelles données --> LSTM + PPO --> Action
```

---

## 8. Points d'Attention Métier

### Ce que le système FAIT

- Prédire la direction des prix (hausse/baisse)
- Prendre des décisions d'achat/vente
- Gérer un portfolio simulé
- Optimiser le ratio rendement/risque

### Ce que le système NE FAIT PAS

- Prédire le prix exact avec certitude
- Garantir des profits
- Prendre en compte les news/événements
- Gérer plusieurs actifs simultanément (portfolio multi-actifs)
- Prendre en compte les slippages réels

### Limitations Connues

1. **Données historiques** : Les performances passées ne prédisent pas les futures
2. **Régimes de marché** : Un modèle entraîné sur un marché haussier peut échouer sur un marché baissier
3. **Overfitting** : Risque de sur-apprentissage sur les données d'entraînement
4. **Liquidité** : Non prise en compte dans la simulation
5. **Gap de prix** : Les sauts de prix overnight ne sont pas simulés

---

## 9. Indicateurs Techniques -- Détail Métier

### RSI (Relative Strength Index)

```
RSI = 100 - (100 / (1 + RS))
RS = moyenne_gain / moyenne_perte (14 jours)
```

**Interprétation** :
- RSI > 70 : Suracheté -- signal de vente
- RSI < 30 : Survendu -- signal d'achat
- RSI = 50 : Neutre

### MACD (Moving Average Convergence Divergence)

```
MACD = EMA(12) - EMA(26)
Signal = EMA(MACD, 9)
Histogram = MACD - Signal
```

**Interprétation** :
- MACD > Signal : Momentum haussier
- MACD < Signal : Momentum baissier
- Croisements : Signaux d'achat/vente

### EMA (Exponential Moving Average)

```
EMA = prix × k + EMA_précédent × (1 - k)
k = 2 / (période + 1)
```

**Interprétation** :
- Prix > EMA : Tendance haussière
- Prix < EMA : Tendance baissière
- EMA_20 > EMA_50 : Signal positif

### Bandes de Bollinger

```
Milieu = SMA(20)
Supérieur = Milieu + 2 × écart-type
Inférieur = Milieu - 2 × écart-type
Largeur = (Sup - Inf) / Milieu
```

**Interprétation** :
- Largeur élevée : Forte volatilité
- Largeur faible : Faible volatilité (précédant souvent un mouvement)
- Prix touche la bande : Possible retournement

### ATR (Average True Range)

```
TR = max(H-L, |H-C_prec|, |L-C_prec|)
ATR = moyenne(TR, 14)
```

**Utilité** :
- Mesure la volatilité absolue
- Utile pour le sizing des positions
- Aide à placer les stop-loss

---

## 10. Workflow d'Utilisation en Production

### Phase 1 : Entraînement Initial

```bash
# 1. Préparer les données (5 ans d'historique)
python train.py --ticker AAPL --data-path "./Data" --step data

# 2. Entraîner le LSTM (environ 50-100 epochs)
python train.py --ticker AAPL --step lstm

# 3. Entraîner l'agent RL (500k timesteps)
python train.py --ticker AAPL --step rl
```

### Phase 2 : Déploiement API

```bash
# Lancer l'API
uvicorn api:app --reload --port 8000

# Endpoints disponibles
GET /predict?ticker=AAPL&days=14    # Prédictions
GET /strategy?ticker=AAPL          # Signaux
GET /performance?ticker=AAPL       # Métriques
```

### Phase 3 : Monitoring

- Vérifier régulièrement le Sharpe Ratio
- Monitorer le Max Drawdown
- Retrainer périodiquement (nouvelles données)

### Phase 4 : Retraining

```bash
# Mise à jour mensuelle recommandée
python train.py --ticker AAPL --data-path "./Data" --step all
```

---

## 11. Glossaire des Termes

| Terme | Définition Simple |
|-------|-------------------|
| **LSTM** | Réseau de neurones qui "se souvient" du passé lointain |
| **RL** | Apprentissage par essai-erreur avec récompenses |
| **PPO** | Algorithme d'apprentissage stable et efficace |
| **Epoch** | Un passage complet sur les données d'entraînement |
| **Timestep** | Une étape de simulation dans l'environnement |
| **Observation** | Ce que l'agent "voit" du marché |
| **Action** | Décision prise par l'agent |
| **Reward** | "Note" donnée à l'action (positive ou négative) |
| **Policy** | Stratégie de décision de l'agent |
| **Sharpe** | Rendement par unité de risque |
| **Drawdown** | Perte depuis le plus haut historique |
| **Overfitting** | Modèle trop spécialisé sur les données d'entraînement |
| **Data Leakage** | Utilisation d'informations futures dans le passé |
| **Slippage** | Différence entre prix voulu et prix exécuté |

---

## 12. Checklist de Validation

Avant de mettre en production :

- [ ] LSTM Directional Accuracy > 55%
- [ ] Sharpe Ratio > 1 sur le test set
- [ ] Max Drawdown acceptable (< 20%)
- [ ] Win Rate > 50%
- [ ] Performance > Buy & Hold sur test set
- [ ] Pas d'overfitting (train vs test similaire)
- [ ] Frais de transaction activés
- [ ] Split temporel strict respecté

---

## 13. Contacts et Support

Pour toute question sur la logique métier :
- Consulter ce document
- Relire les commentaires dans `env.py` et `train.py`
- Vérifier les métriques dans `models/results_*.json`
