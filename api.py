"""
QuantMind — API FastAPI (fichier unique)
=========================================
Lance : uvicorn api:app --reload --port 8000

Charge automatiquement tous les modèles dans models/ au démarrage.
Si un modèle n'existe pas, les endpoints fonctionnent en mode simulé.
"""

import os, sys, json, pickle, logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("quantmind")

# ─── Chemins ─────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_DIR   = "data/processed"
SEQ_LEN    = 60
FEATURES   = ["Close","Volume","EMA_20","EMA_50","RSI",
              "MACD","MACD_signal","BB_width","ATR","Vol_ratio"]

TICKERS = ["AAPL","TSLA","MSFT","GOOGL","AMZN","NVDA","BTC-USD","ETH-USD"]

# ─── Store global ─────────────────────────────────────────────────────────────
store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Chargement des modeles...")
    for t in TICKERS:
        k = t.replace("-","_")
        # Scaler
        sp = f"{MODELS_DIR}/scaler_{k}.pkl"
        if os.path.exists(sp):
            with open(sp,"rb") as f: store[f"scaler_{k}"] = pickle.load(f)
        # Résultats JSON
        for prefix in ["results","lstm_results","rl_results"]:
            jp = f"{MODELS_DIR}/{prefix}_{k}.json"
            if os.path.exists(jp):
                with open(jp) as f: store[f"{prefix}_{k}"] = json.load(f)
    # LSTM
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        for t in TICKERS:
            k = t.replace("-","_")
            mp = f"{MODELS_DIR}/lstm_{k}.keras"
            if os.path.exists(mp):
                store[f"lstm_{k}"] = tf.keras.models.load_model(mp)
                log.info(f"  LSTM charge : {t}")
    except ImportError:
        log.warning("TensorFlow absent — mode simule")
    # RL
    try:
        from stable_baselines3 import PPO
        for t in TICKERS:
            k = t.replace("-","_")
            ap = f"{MODELS_DIR}/ppo_{k}.zip"
            if os.path.exists(ap):
                store[f"ppo_{k}"] = PPO.load(ap)
                log.info(f"  PPO charge  : {t}")
    except ImportError:
        log.warning("stable-baselines3 absent — mode simule")
    log.info(f"API prete ({len(store)} objets en memoire)")
    yield


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="QuantMind API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_df(ticker: str, period: str = "1Y") -> pd.DataFrame:
    k    = ticker.replace("-","_")
    path = f"{DATA_DIR}/{k}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        log.warning(f"CSV absent pour {ticker}, téléchargement yfinance...")
        pmap = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y"}
        raw = yf.Ticker(ticker).history(period=pmap.get(period,"1y"), auto_adjust=True)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        df = raw[["Open","High","Low","Close","Volume"]]

    # Filtrer par période
    delta = {"1M":30,"3M":90,"6M":180,"1Y":365,"2Y":730,"5Y":1825}.get(period,365)
    cutoff = df.index.max() - pd.Timedelta(days=delta)
    return df[df.index >= cutoff]


def simulate_predictions(df, days):
    """Prédictions simulées quand le modèle LSTM n'est pas disponible."""
    last   = float(df["Close"].iloc[-1])
    vol    = float(df["Close"].pct_change().std())
    preds, price = [], last
    last_date = df.index[-1]
    for i in range(1, days+1):
        price *= 1 + np.random.normal(0.0002, vol)
        d = last_date + pd.Timedelta(days=i)
        while d.weekday() >= 5: d += pd.Timedelta(days=1)
        preds.append({"date": d.strftime("%Y-%m-%d"), "price": round(price, 2)})
    return preds


def rule_based_signals(df):
    """Signaux basés sur règles (fallback sans agent RL)."""
    signals = []
    for date, row in df.iterrows():
        rsi, macd, sig = row.get("RSI",50), row.get("MACD",0), row.get("MACD_signal",0)
        ema20, close   = row.get("EMA_20", row["Close"]), row["Close"]
        buys  = int(rsi<35) + int(macd>sig and macd<0) + int(close<ema20*0.97)
        sells = int(rsi>70) + int(macd<sig and macd>0) + int(close>ema20*1.05)
        if buys >= 2:   action, conf = "BUY",  min(0.5+buys*0.1, 0.95)
        elif sells >= 2: action, conf = "SELL", min(0.5+sells*0.1, 0.95)
        else:            action, conf = "HOLD", 0.5
        signals.append({"date":date.strftime("%Y-%m-%d"),"action":action,
                        "price":round(float(close),2),"confidence":round(conf,2),
                        "rsi":round(float(rsi),1),"macd":round(float(macd),4)})
    return signals


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"name":"QuantMind API","version":"1.0.0",
            "models_loaded":[k for k in store if not k.startswith("results")]}

@app.get("/health")
def health():
    return {"status":"ok","models":len(store)}

@app.get("/tickers")
def tickers():
    return {"tickers":[
        {"symbol":"AAPL","name":"Apple Inc.","type":"stock"},
        {"symbol":"TSLA","name":"Tesla Inc.","type":"stock"},
        {"symbol":"MSFT","name":"Microsoft Corp.","type":"stock"},
        {"symbol":"GOOGL","name":"Alphabet Inc.","type":"stock"},
        {"symbol":"AMZN","name":"Amazon.com","type":"stock"},
        {"symbol":"NVDA","name":"NVIDIA Corp.","type":"stock"},
        {"symbol":"BTC-USD","name":"Bitcoin USD","type":"crypto"},
        {"symbol":"ETH-USD","name":"Ethereum USD","type":"crypto"},
    ]}

@app.get("/data/summary")
def data_summary(ticker: str = "AAPL"):
    try:
        df = load_df(ticker, "1Y")
        l, p = df.iloc[-1], df.iloc[-2]
        chg = float(l["Close"]-p["Close"])
        return {
            "ticker":ticker,"price":round(float(l["Close"]),2),
            "change":round(chg,2),"change_pct":round(chg/float(p["Close"])*100,2),
            "volume":int(l["Volume"]),"high_52w":round(float(df["Close"].tail(252).max()),2),
            "low_52w":round(float(df["Close"].tail(252).min()),2),
            "rsi":round(float(l.get("RSI",50)),1),
            "macd":round(float(l.get("MACD",0)),4),
            "ema_20":round(float(l.get("EMA_20",l["Close"])),2),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/data")
def get_data(ticker: str = "AAPL", period: str = "1Y"):
    try:
        df = load_df(ticker, period)
        cols = ["Open","High","Low","Close","Volume","EMA_20","EMA_50",
                "RSI","MACD","MACD_signal","MACD_hist","BB_upper","BB_lower",
                "BB_mid","ATR","Return"]
        cols = [c for c in cols if c in df.columns]
        records = []
        for date, row in df[cols].iterrows():
            r = {"date": date.strftime("%Y-%m-%d")}
            r.update({k: (round(float(v),4) if pd.notna(v) else None) for k,v in row.items()})
            records.append(r)
        return {"ticker":ticker,"period":period,"count":len(records),"data":records}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/predict")
def predict(ticker: str = "AAPL", days: int = 14):
    if not 1 <= days <= 30:
        raise HTTPException(400, "days doit être entre 1 et 30")
    try:
        df  = load_df(ticker, "1Y")
        k   = ticker.replace("-","_")
        mdl = store.get(f"lstm_{k}")
        scl = store.get(f"scaler_{k}")
        res = store.get(f"results_{k}", store.get(f"lstm_results_{k}", {}))
        metrics = res.get("lstm", {})

        if mdl and scl:
            feats = [f for f in FEATURES if f in df.columns]
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
                preds.append({"date":d.strftime("%Y-%m-%d"),"price":round(float(price),2)})
                new_row = seq[-1].copy(); new_row[0] = p
                seq = np.vstack([seq[1:], new_row])
            model_type = "lstm_trained"
        else:
            preds      = simulate_predictions(df, days)
            model_type = "simulated"

        history = [{"date":d.strftime("%Y-%m-%d"),"close":round(float(r["Close"]),2)}
                   for d,r in df.tail(30).iterrows()]
        return {"ticker":ticker,"days":days,"model_type":model_type,
                "metrics":metrics,"predictions":preds,"history":history,
                "last_price":round(float(df["Close"].iloc[-1]),2)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/strategy")
def strategy(ticker: str = "AAPL", period: str = "3M"):
    try:
        df = load_df(ticker, period)
        k  = ticker.replace("-","_")
        agent = store.get(f"ppo_{k}")
        signals = rule_based_signals(df)   # fallback toujours disponible
        model_type = "rule_based"

        if agent:
            try:
                from env import TradingEnv
                lstm_preds = np.zeros(len(df))
                mdl = store.get(f"lstm_{k}")
                scl = store.get(f"scaler_{k}")
                if mdl and scl:
                    feats  = [f for f in FEATURES if f in df.columns]
                    scaled = scl.transform(df[feats].values)
                    for i in range(SEQ_LEN, len(scaled)):
                        seq  = scaled[i-SEQ_LEN:i].reshape(1, SEQ_LEN, len(feats))
                        p    = mdl.predict(seq, verbose=0)[0][0]
                        dummy = np.zeros((1, len(feats))); dummy[0,0] = p
                        lstm_preds[i] = scl.inverse_transform(dummy)[0,0]
                    lstm_preds[:SEQ_LEN] = df["Close"].values[:SEQ_LEN]
                else:
                    lstm_preds = df["Close"].values

                env  = TradingEnv(df.reset_index(drop=True), lstm_preds)
                obs, _ = env.reset()
                signals, done = [], False
                AMAP = {0:"HOLD",1:"BUY",2:"SELL"}
                while not done:
                    act, _ = agent.predict(obs, deterministic=True)
                    idx    = env.idx
                    if idx < len(df):
                        row  = df.iloc[idx]
                        date = df.index[idx]
                        signals.append({"date":date.strftime("%Y-%m-%d"),
                            "action":AMAP[int(act)],"price":round(float(row["Close"]),2),
                            "confidence":0.78,
                            "rsi":round(float(row.get("RSI",50)),1),
                            "macd":round(float(row.get("MACD",0)),4)})
                    obs, _, done, _, _ = env.step(int(act))
                model_type = "rl_trained"
            except Exception as e:
                log.warning(f"Agent RL erreur : {e} — fallback règles")

        buy  = sum(1 for s in signals if s["action"]=="BUY")
        sell = sum(1 for s in signals if s["action"]=="SELL")
        hold = sum(1 for s in signals if s["action"]=="HOLD")
        return {"ticker":ticker,"period":period,"model_type":model_type,
                "signals":signals[-60:],"summary":{"total":len(signals),
                "buy":buy,"sell":sell,"hold":hold,
                "last_action":signals[-1]["action"] if signals else "HOLD"}}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/performance")
def performance(ticker: str = "AAPL"):
    try:
        k   = ticker.replace("-","_")
        res = store.get(f"results_{k}", store.get(f"rl_results_{k}", {}))
        if res:
            return {"ticker":ticker,"rl_agent":res.get("rl_agent"),
                    "buy_and_hold":res.get("buy_and_hold"),"source":"pre_computed"}
        df   = load_df(ticker, "1Y")
        port = 10000 * (df["Close"] / df["Close"].iloc[0])
        rets = port.pct_change().dropna().values
        bh   = {
            "total_return": round(float((port.iloc[-1]/10000-1)*100),2),
            "sharpe_ratio": round(float(rets.mean()/(rets.std()+1e-10)*np.sqrt(252)),4),
            "max_drawdown": round(float(((port.values - np.maximum.accumulate(port.values))/np.maximum.accumulate(port.values)).min()*100),2),
            "final_balance": round(float(port.iloc[-1]),2),
        }
        return {"ticker":ticker,"rl_agent":None,"buy_and_hold":bh,"source":"computed"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/performance/portfolio")
def portfolio_history(ticker: str = "AAPL", period: str = "1Y"):
    try:
        df   = load_df(ticker, period)
        port = 10000 * (df["Close"] / df["Close"].iloc[0])
        data = [{"date":d.strftime("%Y-%m-%d"),
                 "buy_and_hold":round(float(v),2),
                 "close":round(float(df["Close"].loc[d]),2)}
                for d,v in port.items()]
        return {"ticker":ticker,"period":period,"data":data}
    except Exception as e:
        raise HTTPException(500, str(e))
