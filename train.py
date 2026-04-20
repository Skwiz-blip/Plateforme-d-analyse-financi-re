"""
╔══════════════════════════════════════════════════════════════╗
║           QuantMind — Pipeline d'Entraînement Complet        ║
║           LSTM (Prédiction) + PPO RL (Stratégie)             ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
    # Pipeline complet (recommandé)
    python train.py --ticker AAPL --data-path "C:/chemin/vers/Data"

    # Étapes séparées
    python train.py --ticker AAPL --data-path "C:/chemin/vers/Data" --step data
    python train.py --ticker AAPL --step lstm
    python train.py --ticker AAPL --step rl
    python train.py --ticker AAPL --step all   (défaut)

    # Plusieurs tickers
    python train.py --all-tickers --data-path "C:/chemin/vers/Data"

DATASET KAGGLE:
    Télécharge depuis :
    https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
    Le dossier Data/ doit contenir Stocks/ et ETFs/
    Format fichier : aapl.us.txt (Date,Open,High,Low,Close,Volume,OpenInt)

OUTPUTS:
    models/lstm_{TICKER}.keras       ← Réseau LSTM entraîné
    models/scaler_{TICKER}.pkl       ← Normalisation (MinMaxScaler)
    models/ppo_{TICKER}.zip          ← Agent RL entraîné
    models/results_{TICKER}.json     ← Métriques LSTM + RL
    data/processed/{TICKER}.csv      ← Données + indicateurs techniques
"""

# ─── Imports ─────────────────────────────────────────────────────────────────
import os
import sys
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── Configuration globale ───────────────────────────────────────────────────

MODELS_DIR    = "models"
DATA_RAW_DIR  = "data/raw"
DATA_PROC_DIR = "data/processed"

for d in [MODELS_DIR, DATA_RAW_DIR, DATA_PROC_DIR]:
    os.makedirs(d, exist_ok=True)

SEQ_LEN  = 60           # Fenêtre temporelle LSTM (jours)
FEATURES = [            # Features utilisées pour LSTM + RL
    "Close", "Volume",
    "EMA_20", "EMA_50",
    "RSI", "MACD", "MACD_signal",
    "BB_width", "ATR", "Vol_ratio",
]

TICKERS_KAGGLE = {
    "AAPL":    "aapl.us.txt",
    "TSLA":    "tsla.us.txt",
    "MSFT":    "msft.us.txt",
    "GOOGL":   "googl.us.txt",
    "AMZN":    "amzn.us.txt",
    "NVDA":    "nvda.us.txt",
    "BTC-USD": None,   # yfinance only
    "ETH-USD": None,   # yfinance only
}


# ══════════════════════════════════════════════════════════════════════════════
#  PARTIE 1 — DONNÉES : Chargement + Indicateurs techniques
# ══════════════════════════════════════════════════════════════════════════════

def plog(msg, indent=0):
    """Print log message with optional indentation."""
    prefix = "  " * indent
    print(f"{prefix}{msg}")


def load_kaggle_file(ticker: str, data_path: str) -> pd.DataFrame | None:
    """Lit le fichier .txt du dataset Kaggle."""
    filename = TICKERS_KAGGLE.get(ticker)
    if not filename:
        return None

    # Chercher dans Stocks/ puis racine
    candidates = [
        os.path.join(data_path, "Stocks", filename),
        os.path.join(data_path, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(
                path, parse_dates=["Date"], index_col="Date",
                dtype={"Open": float, "High": float, "Low": float,
                       "Close": float, "Volume": float}
            )
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.sort_index(inplace=True)
            df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
            return df

    plog(f"⚠  Fichier Kaggle introuvable : {filename}", 2)
    return None


def fetch_yfinance(ticker: str, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Complète ou remplace avec yfinance."""
    try:
        t = yf.Ticker(ticker)
        if since is not None:
            df = t.history(start=since, auto_adjust=True)
        else:
            df = t.history(period="5y", auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    except Exception as e:
        plog(f"⚠  yfinance erreur : {e}", 2)
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule tous les indicateurs techniques nécessaires au LSTM et au RL :
      - EMA 20, EMA 50
      - RSI (14 périodes)
      - MACD (12, 26, 9) + signal + histogramme
      - Bandes de Bollinger (20 jours, 2σ) + largeur normalisée
      - ATR (14 périodes)
      - Returns journaliers et log-returns
      - Ratio volume / moyenne mobile 20j
    """
    df = df.copy()

    # EMA
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands (20, 2σ)
    rm  = df["Close"].rolling(20).mean()
    rs  = df["Close"].rolling(20).std()
    df["BB_upper"] = rm + 2 * rs
    df["BB_lower"] = rm - 2 * rs
    df["BB_mid"]   = rm
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (rm + 1e-10)

    # ATR (14)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Returns
    df["Return"]     = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Volume normalisé
    df["Vol_MA20"]  = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / (df["Vol_MA20"] + 1e-10)

    df.dropna(inplace=True)
    return df


def step_data(ticker: str, data_path: str | None) -> pd.DataFrame:
    """
    ÉTAPE 1 — Chargement et préparation des données.

    Priorité :
      1. Fichier Kaggle (historique long, jusqu'en 2017)
      2. yfinance (complète avec données récentes / crypto)
    """
    plog(f"\n{'─'*55}")
    plog(f"ÉTAPE 1 — Données : {ticker}")
    plog(f"{'─'*55}")

    kaggle_df = None
    if data_path:
        kaggle_df = load_kaggle_file(ticker, data_path)
        if kaggle_df is not None:
            plog(f"✓ Kaggle  : {len(kaggle_df)} jours "
                f"({kaggle_df.index.min().date()} → {kaggle_df.index.max().date()})", 1)

    # Compléter / remplacer avec yfinance
    since = kaggle_df.index.max() if kaggle_df is not None else None
    yf_df = fetch_yfinance(ticker, since)

    if len(yf_df) > 0:
        plog(f"✓ yfinance: {len(yf_df)} jours de plus", 1)
        if kaggle_df is not None:
            combined = pd.concat([kaggle_df, yf_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = yf_df
    elif kaggle_df is not None:
        combined = kaggle_df
        plog("⚠  yfinance indisponible — données Kaggle seules", 1)
    else:
        raise RuntimeError(f"Aucune donnée trouvée pour {ticker}")

    # Garder 5 ans max
    cutoff  = combined.index.max() - pd.DateOffset(years=5)
    combined = combined[combined.index >= cutoff]

    # Sauvegarder brut
    raw_path = os.path.join(DATA_RAW_DIR, f"{ticker.replace('-','_')}.csv")
    combined.to_csv(raw_path)

    # Calculer les indicateurs
    plog("→ Calcul des indicateurs techniques...", 1)
    df_proc = compute_indicators(combined)

    proc_path = os.path.join(DATA_PROC_DIR, f"{ticker.replace('-','_')}.csv")
    df_proc.to_csv(proc_path)

    plog(f"✓ Total   : {len(df_proc)} jours avec {len(df_proc.columns)} features", 1)
    plog(f"✓ Sauvegardé → {proc_path}", 1)
    return df_proc


# ══════════════════════════════════════════════════════════════════════════════
#  PARTIE 2 — LSTM : Architecture + Entraînement + Évaluation
# ══════════════════════════════════════════════════════════════════════════════

def build_sequences(data: np.ndarray):
    """Crée les séquences glissantes (window = SEQ_LEN → cible J+1)."""
    X, y = [], []
    for i in range(SEQ_LEN, len(data)):
        X.append(data[i - SEQ_LEN:i])
        y.append(data[i, 0])     # Close = colonne 0
    return np.array(X), np.array(y)


def split_temporal(X, y, train=0.80, val=0.10):
    """Split temporel strict — pas de shuffle pour éviter le data leakage."""
    n   = len(X)
    t1  = int(n * train)
    t2  = int(n * (train + val))
    return (X[:t1], y[:t1]), (X[t1:t2], y[t1:t2]), (X[t2:], y[t2:])


def build_lstm(seq_len: int, n_features: int):
    """
    Architecture LSTM :
      128 → Dropout 0.2
      64  → Dropout 0.2
      32  → Dropout 0.2
      BatchNorm → Dense 32 → Dense 1
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization, Input
    )
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model


def inv_close(values: np.ndarray, scaler, n_features: int) -> np.ndarray:
    """Inverse-transform uniquement la colonne Close (index 0)."""
    dummy = np.zeros((len(values), n_features))
    dummy[:, 0] = values
    return scaler.inverse_transform(dummy)[:, 0]


def directional_accuracy(y_true, y_pred) -> float:
    """% de fois où la direction (hausse/baisse) est correctement prédite."""
    real_dir  = np.diff(y_true)
    pred_dir  = np.diff(y_pred)
    return (np.sign(real_dir) == np.sign(pred_dir)).mean() * 100


def step_lstm(ticker: str, df: pd.DataFrame | None = None) -> dict:
    """
    ÉTAPE 2 — Entraînement du LSTM.

    Produit :
      models/lstm_{ticker}.h5      ← meilleur modèle (EarlyStopping)
      models/scaler_{ticker}.pkl   ← scaler fitté sur train uniquement
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )
    from tensorflow.keras.models import load_model

    plog(f"\n{'─'*55}")
    plog(f"ÉTAPE 2 — LSTM : {ticker}")
    plog(f"{'─'*55}")

    # Charger les données si non fournies
    if df is None:
        proc_path = os.path.join(DATA_PROC_DIR, f"{ticker.replace('-','_')}.csv")
        if not os.path.exists(proc_path):
            raise FileNotFoundError(
                f"Données introuvables pour {ticker}.\n"
                "Lance d'abord : python train.py --ticker {ticker} --step data"
            )
        df = pd.read_csv(proc_path, index_col="Date", parse_dates=True)

    feats = [f for f in FEATURES if f in df.columns]
    n_features = len(feats)
    data = df[feats].values
    plog(f"→ Features ({n_features}) : {', '.join(feats)}", 1)

    # Normalisation — fit uniquement sur 80% train pour éviter data leakage
    train_end = int(len(data) * 0.80)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_end])
    data_scaled = scaler.transform(data)

    scaler_path = os.path.join(MODELS_DIR, f"scaler_{ticker.replace('-','_')}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    plog(f"✓ Scaler sauvegardé → {scaler_path}", 1)

    # Séquences + split
    X, y = build_sequences(data_scaled)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_temporal(X, y)
    plog(f"✓ Train:{X_tr.shape}  Val:{X_val.shape}  Test:{X_te.shape}", 1)

    # Modèle
    model      = build_lstm(SEQ_LEN, n_features)
    model_path = os.path.join(MODELS_DIR, f"lstm_{ticker.replace('-','_')}.keras")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=6, min_lr=1e-6, verbose=1),
    ]

    plog("\n→ Entraînement LSTM (max 100 epochs, EarlyStopping patience=12)...\n", 1)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=32,
        callbacks=callbacks, verbose=1,
    )

    # Évaluation
    model = load_model(model_path)
    y_pred_sc = model.predict(X_te, verbose=0).flatten()
    y_te_real  = inv_close(y_te,       scaler, n_features)
    y_pred_real = inv_close(y_pred_sc, scaler, n_features)

    rmse  = float(np.sqrt(mean_squared_error(y_te_real, y_pred_real)))
    mae   = float(mean_absolute_error(y_te_real, y_pred_real))
    mape  = float(np.mean(np.abs((y_te_real - y_pred_real) / (y_te_real + 1e-10))) * 100)
    da    = float(directional_accuracy(y_te_real, y_pred_real))
    epochs_done = len(history.history["loss"])

    metrics = {
        "ticker": ticker,
        "lstm": {
            "rmse": round(rmse, 4),
            "mae":  round(mae, 4),
            "mape": round(mape, 4),
            "directional_accuracy": round(da, 2),
            "epochs": epochs_done,
            "train_samples": int(X_tr.shape[0]),
            "test_samples":  int(X_te.shape[0]),
        },
        "features": feats,
        "seq_len": SEQ_LEN,
    }

    plog(f"\n{'─'*40}", 1)
    plog("MÉTRIQUES LSTM", 1)
    plog(f"{'─'*40}", 1)
    plog(f"  RMSE                 : ${rmse:.2f}", 1)
    plog(f"  MAE                  : ${mae:.2f}", 1)
    plog(f"  MAPE                 : {mape:.2f}%", 1)
    plog(f"  Directional Accuracy : {da:.1f}%  ← métrique clé", 1)
    plog(f"  Epochs               : {epochs_done}", 1)
    plog(f"✓ Modèle sauvegardé → {model_path}", 1)

    return metrics, y_pred_real, y_te_real, df


# ══════════════════════════════════════════════════════════════════════════════
#  PARTIE 3 — RL : Environnement Gym Custom + Agent PPO + Backtesting
# ══════════════════════════════════════════════════════════════════════════════

from env import TradingEnv, GYM_OK


def buy_and_hold_metrics(df: pd.DataFrame, initial: float = 10_000.0) -> dict:
    """Stratégie de référence : acheter au début, vendre à la fin."""
    p0, pf = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
    shares  = initial // p0
    final   = shares * pf + (initial - shares * p0)
    portfolio = initial * (df["Close"] / p0)
    returns   = portfolio.pct_change().dropna().values
    sharpe    = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252)
    rm        = np.maximum.accumulate(portfolio.values)
    max_dd    = ((portfolio.values - rm) / rm).min() * 100
    return {
        "total_return":  round(float((final / initial - 1) * 100), 2),
        "sharpe_ratio":  round(float(sharpe), 4),
        "max_drawdown":  round(float(max_dd), 2),
        "final_balance": round(float(final), 2),
        "win_rate":      None,
        "n_trades":      1,
    }


def get_lstm_predictions(df: pd.DataFrame, ticker: str) -> np.ndarray:
    """
    Génère les prédictions LSTM sur tout le dataset.
    Utilisées comme feature d'entrée de l'agent RL (couplage).
    """
    model_path  = os.path.join(MODELS_DIR, f"lstm_{ticker.replace('-','_')}.keras")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{ticker.replace('-','_')}.pkl")

    preds = df["Close"].values.copy()   # fallback : prix réels

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        plog("⚠  Modèle LSTM introuvable — RL utilisera les prix réels", 2)
        return preds

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        feats = [f for f in FEATURES if f in df.columns]
        scaled = scaler.transform(df[feats].values)

        for i in range(SEQ_LEN, len(scaled)):
            seq = scaled[i - SEQ_LEN:i].reshape(1, SEQ_LEN, len(feats))
            p   = model.predict(seq, verbose=0)[0][0]
            dummy = np.zeros((1, len(feats)))
            dummy[0, 0] = p
            preds[i] = scaler.inverse_transform(dummy)[0, 0]

        plog(f"✓ {len(scaled) - SEQ_LEN} prédictions LSTM générées pour RL", 2)
    except Exception as e:
        plog(f"⚠  Erreur LSTM pour RL : {e} — utilisation des prix réels", 2)

    return preds


def step_rl(ticker: str, df: pd.DataFrame | None = None) -> dict:
    """
    ÉTAPE 3 — Entraînement de l'agent RL (PPO).

    Produit :
      models/ppo_{ticker}.zip   ← agent entraîné
    Évalue la stratégie RL vs Buy & Hold.
    """
    plog(f"\n{'─'*55}")
    plog(f"ÉTAPE 3 — RL / PPO : {ticker}")
    plog(f"{'─'*55}")

    if not GYM_OK:
        plog("✗ gymnasium non disponible — installe : pip install gymnasium", 1)
        return {}

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        plog("✗ stable-baselines3 non disponible — installe : pip install stable-baselines3", 1)
        return {}

    # Charger les données
    if df is None:
        proc_path = os.path.join(DATA_PROC_DIR, f"{ticker.replace('-','_')}.csv")
        df = pd.read_csv(proc_path, index_col="Date", parse_dates=True)

    # Split 80/20 temporel
    split    = int(len(df) * 0.80)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_test  = df.iloc[split:].reset_index(drop=True)
    plog(f"✓ Train: {len(df_train)} jours | Test: {len(df_test)} jours", 1)

    # Prédictions LSTM → features de l'agent
    plog("→ Génération des prédictions LSTM pour l'agent...", 1)
    lstm_train = get_lstm_predictions(df_train, ticker)
    lstm_test  = get_lstm_predictions(df_test,  ticker)

    # Vérifier l'environnement
    plog("→ Vérification de l'environnement Gym...", 1)
    check_env(TradingEnv(df_train.head(120), lstm_train[:120]), warn=True)
    plog("✓ Environnement Gym valide", 1)

    # Entraînement PPO
    train_env = DummyVecEnv([lambda: TradingEnv(df_train, lstm_train)])
    eval_env  = DummyVecEnv([lambda: TradingEnv(df_test,  lstm_test)])

    agent_path = os.path.join(MODELS_DIR, f"ppo_{ticker.replace('-','_')}")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=MODELS_DIR,
        eval_freq=10_000,
        n_eval_episodes=3,
        verbose=0,
    )

    agent = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
    )

    plog("\n→ Entraînement PPO (500 000 timesteps)...\n", 1)
    agent.learn(total_timesteps=500_000, callback=eval_cb)
    agent.save(agent_path)
    plog(f"\n✓ Agent PPO sauvegardé → {agent_path}.zip", 1)

    # Backtesting
    plog("\n→ Backtesting sur le test set...", 1)
    test_env = TradingEnv(df_test, lstm_test)
    obs, _  = test_env.reset()
    done     = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(int(action))

    rl_m = test_env.metrics()
    bh_m = buy_and_hold_metrics(df_test)

    plog(f"\n{'─'*40}", 1)
    plog("BACKTESTING — RL vs Buy & Hold", 1)
    plog(f"{'─'*40}", 1)
    plog(f"  {'Métrique':<25} {'Agent RL':>10} {'Buy&Hold':>10}", 1)
    plog(f"  {'─'*45}", 1)
    for k, lbl in [("total_return","Total Return (%)"),
                    ("sharpe_ratio","Sharpe Ratio"),
                    ("max_drawdown","Max Drawdown (%)")]:
        plog(f"  {lbl:<25} {rl_m[k]:>10.2f} {bh_m[k]:>10.2f}", 1)
    plog(f"  {'Win Rate (%)':<25} {rl_m['win_rate']:>10.1f} {'—':>10}", 1)
    plog(f"  {'Nb Trades':<25} {rl_m['n_trades']:>10} {bh_m['n_trades']:>10}", 1)

    return {"rl": rl_m, "buy_and_hold": bh_m}


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(ticker: str, data_path: str | None, step: str):
    """Lance les étapes demandées et sauvegarde les résultats."""
    results = {"ticker": ticker, "timestamp": datetime.now().isoformat()}
    df      = None

    # ─── Étape Données ───────────────────────────────────────────────────────
    if step in ("all", "data"):
        df = step_data(ticker, data_path)

    # ─── Charger si déjà fait ────────────────────────────────────────────────
    if df is None and step in ("lstm", "rl"):
        proc_path = os.path.join(DATA_PROC_DIR, f"{ticker.replace('-','_')}.csv")
        if not os.path.exists(proc_path):
            plog(f"✗ Données introuvables. Lance d'abord :")
            plog(f"  python train.py --ticker {ticker} --step data")
            sys.exit(1)
        df = pd.read_csv(proc_path, index_col="Date", parse_dates=True)

    # ─── Étape LSTM ──────────────────────────────────────────────────────────
    if step in ("all", "lstm"):
        metrics, _, _, _ = step_lstm(ticker, df)
        results.update(metrics)

    # ─── Étape RL ────────────────────────────────────────────────────────────
    if step in ("all", "rl"):
        rl_results = step_rl(ticker, df)
        if rl_results:
            results["rl_agent"]    = rl_results["rl"]
            results["buy_and_hold"] = rl_results["buy_and_hold"]

    # ─── Sauvegarder tous les résultats ──────────────────────────────────────
    results_path = os.path.join(MODELS_DIR, f"results_{ticker.replace('-','_')}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    plog(f"\n{'═'*55}")
    plog(f"✓ Pipeline terminé pour {ticker}")
    plog(f"  Résultats → {results_path}")
    plog(f"  Modèles   → {MODELS_DIR}/")
    plog(f"{'═'*55}\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QuantMind — Entraînement LSTM + RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python train.py --ticker AAPL --data-path "C:/Users/moi/Downloads/Data"
  python train.py --ticker AAPL --step lstm
  python train.py --ticker AAPL --step rl
  python train.py --all-tickers --data-path "/home/moi/Data"
        """
    )
    parser.add_argument("--ticker",      type=str, default="AAPL",
                        help="Ticker (défaut: AAPL)")
    parser.add_argument("--data-path",   type=str, default=None,
                        help='Chemin vers le dossier Data/ du dataset Kaggle')
    parser.add_argument("--step",        type=str, default="all",
                        choices=["all", "data", "lstm", "rl"],
                        help="Étape à exécuter (défaut: all)")
    parser.add_argument("--all-tickers", action="store_true",
                        help="Traiter tous les tickers disponibles")
    args = parser.parse_args()

    print("\n" + "╔" + "═"*53 + "╗")
    print("║" + "  QuantMind — Pipeline d'Entraînement".center(53) + "║")
    print("║" + "  LSTM (Prédiction) + PPO RL (Stratégie)".center(53) + "║")
    print("╚" + "═"*53 + "╝")

    tickers = list(TICKERS_KAGGLE.keys()) if args.all_tickers else [args.ticker]

    for ticker in tickers:
        try:
            run_pipeline(ticker, args.data_path, args.step)
        except Exception as e:
            plog(f"\n✗ Erreur pour {ticker} : {e}")
            if len(tickers) == 1:
                raise


if __name__ == "__main__":
    main()
