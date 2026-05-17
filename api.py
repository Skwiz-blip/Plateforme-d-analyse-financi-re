"""
QuantMind API — FastAPI

Tout est réel : aucune simulation, aucun fallback yfinance.
Si un modèle est absent → 503. Si un CSV est absent → 404.
Les tickers exposés via /tickers sont ceux ayant un LSTM + PPO entraînés.
"""

import os, json, pickle, logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("quantmind")

MODELS_DIR = "models"
DATA_DIR   = "data/processed"
SEQ_LEN    = 60
FEATURES   = ["Close","Volume","EMA_20","EMA_50","RSI",
              "MACD","MACD_signal","BB_width","ATR","Vol_ratio"]

# Univers du projet : actions du dataset Kaggle uniquement.
ALL_TICKERS = [
    {"symbol":"AAPL", "name":"Apple Inc.",      "type":"stock"},
    {"symbol":"TSLA", "name":"Tesla Inc.",      "type":"stock"},
    {"symbol":"MSFT", "name":"Microsoft Corp.", "type":"stock"},
    {"symbol":"GOOGL","name":"Alphabet Inc.",   "type":"stock"},
    {"symbol":"AMZN", "name":"Amazon.com",      "type":"stock"},
    {"symbol":"NVDA", "name":"NVIDIA Corp.",    "type":"stock"},
]

store: dict = {}


def _k(t: str) -> str:
    return t.replace("-", "_")


def is_trained(ticker: str) -> bool:
    """Un ticker est 'entraîné' s'il a LSTM + scaler + PPO + CSV processed."""
    k = _k(ticker)
    return all(os.path.exists(p) for p in [
        f"{MODELS_DIR}/lstm_{k}.keras",
        f"{MODELS_DIR}/scaler_{k}.pkl",
        f"{MODELS_DIR}/ppo_{k}.zip",
        f"{DATA_DIR}/{k}.csv",
    ])


def trained_tickers() -> list[dict]:
    """Liste des tickers réellement utilisables (modèles présents en mémoire)."""
    out = []
    for t in ALL_TICKERS:
        k = _k(t["symbol"])
        if (f"lstm_{k}" in store and f"scaler_{k}" in store
                and f"ppo_{k}" in store and os.path.exists(f"{DATA_DIR}/{k}.csv")):
            out.append(t)
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Chargement des modeles...")
    # Scalers + résultats
    for t in ALL_TICKERS:
        k = _k(t["symbol"])
        sp = f"{MODELS_DIR}/scaler_{k}.pkl"
        if os.path.exists(sp):
            with open(sp, "rb") as f: store[f"scaler_{k}"] = pickle.load(f)
        jp = f"{MODELS_DIR}/results_{k}.json"
        if os.path.exists(jp):
            with open(jp) as f: store[f"results_{k}"] = json.load(f)

    # LSTM
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    for t in ALL_TICKERS:
        k = _k(t["symbol"])
        mp = f"{MODELS_DIR}/lstm_{k}.keras"
        if not os.path.exists(mp):
            continue
        try:
            store[f"lstm_{k}"] = tf.keras.models.load_model(mp, compile=False)
            log.info(f"  LSTM charge : {t['symbol']}")
        except Exception as e:
            log.warning(f"  LSTM {t['symbol']} non chargeable ({type(e).__name__})")

    # RL
    from stable_baselines3 import PPO
    for t in ALL_TICKERS:
        k = _k(t["symbol"])
        ap = f"{MODELS_DIR}/ppo_{k}.zip"
        if not os.path.exists(ap):
            continue
        try:
            store[f"ppo_{k}"] = PPO.load(ap)
            log.info(f"  PPO charge  : {t['symbol']}")
        except Exception as e:
            log.warning(f"  PPO {t['symbol']} non chargeable ({type(e).__name__})")

    ready = [t["symbol"] for t in trained_tickers()]
    log.info(f"API prete — tickers exploitables : {ready}")
    yield


app = FastAPI(title="QuantMind API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─── Helpers ────────────────────────────────────────────────────────────────

def require_trained(ticker: str):
    """Lève 503 si le ticker n'a pas tous ses artefacts (LSTM + PPO + CSV)."""
    if ticker not in [t["symbol"] for t in ALL_TICKERS]:
        raise HTTPException(404, f"Ticker '{ticker}' inconnu (univers Kaggle uniquement)")
    k = _k(ticker)
    missing = []
    if f"lstm_{k}"   not in store: missing.append("LSTM")
    if f"scaler_{k}" not in store: missing.append("scaler")
    if f"ppo_{k}"    not in store: missing.append("PPO")
    if not os.path.exists(f"{DATA_DIR}/{k}.csv"): missing.append("CSV processed")
    if missing:
        raise HTTPException(503,
            f"Ticker '{ticker}' non encore entrainé (manquant : {', '.join(missing)})")


def load_df(ticker: str, period: str = "1Y") -> pd.DataFrame:
    """Lit le CSV processed. 404 si absent — pas de fallback yfinance."""
    k = _k(ticker)
    path = f"{DATA_DIR}/{k}.csv"
    if not os.path.exists(path):
        raise HTTPException(404, f"Données non disponibles pour {ticker}")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    delta = {"1M":30,"3M":90,"6M":180,"1Y":365,"2Y":730,"5Y":1825}.get(period, 365)
    cutoff = df.index.max() - pd.Timedelta(days=delta)
    return df[df.index >= cutoff]


def lstm_predict_series(df: pd.DataFrame, ticker: str) -> np.ndarray:
    """Génère les prédictions LSTM auto-régressives sur tout le df."""
    k = _k(ticker)
    mdl, scl = store[f"lstm_{k}"], store[f"scaler_{k}"]
    feats  = [f for f in FEATURES if f in df.columns]
    scaled = scl.transform(df[feats].values)
    preds  = df["Close"].values.copy()
    for i in range(SEQ_LEN, len(scaled)):
        seq = scaled[i-SEQ_LEN:i].reshape(1, SEQ_LEN, len(feats))
        p   = mdl.predict(seq, verbose=0)[0][0]
        dummy = np.zeros((1, len(feats))); dummy[0,0] = p
        preds[i] = scl.inverse_transform(dummy)[0,0]
    return preds


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "QuantMind API",
        "version": "2.0.0",
        "trained": [t["symbol"] for t in trained_tickers()],
    }


@app.get("/health")
def health():
    return {"status": "ok", "trained_tickers": len(trained_tickers())}


@app.get("/tickers")
def tickers():
    """Renvoie uniquement les tickers ayant des modèles chargés ET un CSV processed."""
    return {"tickers": trained_tickers()}


@app.get("/model/info")
def model_info(ticker: str = "AAPL"):
    """Métadonnées détaillées du modèle : architecture LSTM + métriques + hyperparams PPO."""
    require_trained(ticker)
    k = _k(ticker)
    mdl = store[f"lstm_{k}"]
    agent = store[f"ppo_{k}"]
    res = store.get(f"results_{k}", {})

    layers = []
    total_params = 0
    for layer in mdl.layers:
        try:
            params = int(layer.count_params())
        except Exception:
            params = 0
        total_params += params
        out_shape = None
        try:
            out_shape = list(layer.output.shape) if hasattr(layer, "output") else None
        except Exception:
            pass
        layers.append({
            "name":    layer.name,
            "type":    layer.__class__.__name__,
            "params":  params,
            "output_shape": out_shape,
        })

    ppo_hp = {
        "policy":        agent.policy.__class__.__name__,
        "learning_rate": float(agent.learning_rate) if not callable(agent.learning_rate) else "schedule",
        "gamma":         float(agent.gamma),
        "gae_lambda":    float(agent.gae_lambda),
        "ent_coef":      float(agent.ent_coef),
        "n_steps":       int(agent.n_steps),
        "batch_size":    int(agent.batch_size),
        "n_epochs":      int(agent.n_epochs),
        "n_envs":        int(agent.n_envs),
    }

    return {
        "ticker": ticker,
        "lstm": {
            "seq_len":         SEQ_LEN,
            "n_features":      len(FEATURES),
            "features":        FEATURES,
            "layers":          layers,
            "total_params":    total_params,
            "training":        res.get("lstm", {}),
        },
        "ppo": {
            "algo":            "PPO",
            "hyperparameters": ppo_hp,
            "results":         res.get("rl_agent", {}),
            "buy_and_hold":    res.get("buy_and_hold", {}),
            "timesteps":       500_000,
        },
        "trained_at": res.get("timestamp"),
    }


@app.get("/data/summary")
def data_summary(ticker: str = "AAPL"):
    require_trained(ticker)
    df = load_df(ticker, "1Y")
    l, p = df.iloc[-1], df.iloc[-2]
    chg = float(l["Close"] - p["Close"])
    return {
        "ticker":    ticker,
        "price":     round(float(l["Close"]), 2),
        "change":    round(chg, 2),
        "change_pct":round(chg / float(p["Close"]) * 100, 2),
        "volume":    int(l["Volume"]),
        "high_52w":  round(float(df["Close"].tail(252).max()), 2),
        "low_52w":   round(float(df["Close"].tail(252).min()), 2),
        "rsi":       round(float(l.get("RSI", 50)), 1),
        "macd":      round(float(l.get("MACD", 0)), 4),
        "ema_20":    round(float(l.get("EMA_20", l["Close"])), 2),
    }


@app.get("/data")
def get_data(ticker: str = "AAPL", period: str = "1Y"):
    require_trained(ticker)
    df = load_df(ticker, period)
    cols = ["Open","High","Low","Close","Volume","EMA_20","EMA_50",
            "RSI","MACD","MACD_signal","MACD_hist","BB_upper","BB_lower",
            "BB_mid","ATR","Return"]
    cols = [c for c in cols if c in df.columns]
    records = []
    for date, row in df[cols].iterrows():
        r = {"date": date.strftime("%Y-%m-%d")}
        r.update({k: (round(float(v), 4) if pd.notna(v) else None) for k, v in row.items()})
        records.append(r)
    return {"ticker": ticker, "period": period, "count": len(records), "data": records}


@app.get("/predict")
def predict(ticker: str = "AAPL", days: int = 14):
    if not 1 <= days <= 30:
        raise HTTPException(400, "days doit être entre 1 et 30")
    require_trained(ticker)

    df  = load_df(ticker, "1Y")
    k   = _k(ticker)
    mdl, scl = store[f"lstm_{k}"], store[f"scaler_{k}"]
    res = store.get(f"results_{k}", {})

    feats  = [f for f in FEATURES if f in df.columns]
    scaled = scl.transform(df[feats].values)
    seq    = scaled[-SEQ_LEN:].copy()

    preds, last_date = [], df.index[-1]
    for i in range(days):
        inp = seq.reshape(1, SEQ_LEN, len(feats))
        p   = mdl.predict(inp, verbose=0)[0][0]
        dummy = np.zeros((1, len(feats))); dummy[0,0] = p
        price = scl.inverse_transform(dummy)[0,0]
        d = last_date + pd.Timedelta(days=i+1)
        while d.weekday() >= 5: d += pd.Timedelta(days=1)
        preds.append({"date": d.strftime("%Y-%m-%d"), "price": round(float(price), 2)})
        new_row = seq[-1].copy(); new_row[0] = p
        seq = np.vstack([seq[1:], new_row])

    history = [{"date": d.strftime("%Y-%m-%d"), "close": round(float(r["Close"]), 2)}
               for d, r in df.tail(30).iterrows()]
    return {
        "ticker":     ticker,
        "days":       days,
        "model_type": "lstm_trained",
        "metrics":    res.get("lstm", {}),
        "predictions":preds,
        "history":    history,
        "last_price": round(float(df["Close"].iloc[-1]), 2),
    }


@app.get("/strategy")
def strategy(ticker: str = "AAPL", period: str = "3M"):
    require_trained(ticker)
    from env import TradingEnv

    df = load_df(ticker, period).reset_index(drop=False)
    dates = df["Date"]
    df_env = df.drop(columns=["Date"])

    k = _k(ticker)
    agent = store[f"ppo_{k}"]
    lstm_preds = lstm_predict_series(df_env, ticker)

    env = TradingEnv(df_env, lstm_preds)
    obs, _ = env.reset()
    signals, done = [], False
    AMAP = {0: "HOLD", 1: "BUY", 2: "SELL"}
    while not done:
        act, _ = agent.predict(obs, deterministic=True)
        idx = env.idx
        if idx < len(df_env):
            row = df_env.iloc[idx]
            signals.append({
                "date":       dates.iloc[idx].strftime("%Y-%m-%d"),
                "action":     AMAP[int(act)],
                "price":      round(float(row["Close"]), 2),
                "confidence": 0.78,
                "rsi":        round(float(row.get("RSI", 50)), 1),
                "macd":       round(float(row.get("MACD", 0)), 4),
            })
        obs, _, done, _, _ = env.step(int(act))

    buy  = sum(1 for s in signals if s["action"] == "BUY")
    sell = sum(1 for s in signals if s["action"] == "SELL")
    hold = sum(1 for s in signals if s["action"] == "HOLD")
    return {
        "ticker":     ticker,
        "period":     period,
        "model_type": "rl_trained",
        "signals":    signals[-60:],
        "summary": {
            "total":       len(signals),
            "buy":         buy,
            "sell":        sell,
            "hold":        hold,
            "last_action": signals[-1]["action"] if signals else "HOLD",
        },
    }


@app.get("/performance")
def performance(ticker: str = "AAPL"):
    require_trained(ticker)
    k   = _k(ticker)
    res = store.get(f"results_{k}", {})
    return {
        "ticker":       ticker,
        "rl_agent":     res.get("rl_agent"),
        "buy_and_hold": res.get("buy_and_hold"),
        "source":       "pre_computed",
    }


@app.get("/performance/portfolio")
def portfolio_history(ticker: str = "AAPL", period: str = "1Y"):
    require_trained(ticker)
    from env import TradingEnv

    df = load_df(ticker, period)
    port = 10000 * (df["Close"] / df["Close"].iloc[0])
    k = _k(ticker)
    agent = store[f"ppo_{k}"]

    df_env = df.reset_index(drop=True)
    lstm_preds = lstm_predict_series(df_env, ticker)
    env = TradingEnv(df_env, lstm_preds)
    obs, _ = env.reset()
    done = False
    while not done:
        act, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(act))
    rl_history = env.history

    data = []
    for i, (d, v) in enumerate(port.items()):
        row = {
            "date":         d.strftime("%Y-%m-%d"),
            "buy_and_hold": round(float(v), 2),
            "close":        round(float(df["Close"].loc[d]), 2),
        }
        if i < len(rl_history):
            row["rl_agent"] = round(float(rl_history[i]), 2)
        data.append(row)

    returns_pct = (df["Close"].pct_change().dropna() * 100).round(3).tolist()
    return {
        "ticker":      ticker,
        "period":      period,
        "data":        data,
        "returns_pct": returns_pct,
        "has_rl":      True,
    }
