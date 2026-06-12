import os
import sys
import json
import time
import pickle
import logging
from collections import deque

import numpy as np

# Le consumer est lancé depuis la racine du projet (local et Docker)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import FEATURES, SEQ_LEN, TARGET_SCALE, MODELS_DIR  # noqa: E402

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("consumer")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
GROUP_ID        = os.getenv("GROUP_ID", "quantmind-consumer")

TOPIC_IN  = "prices"
TOPIC_OUT = "signals"
AMAP      = {0: "HOLD", 1: "BUY", 2: "SELL"}

 

class PaperTrader:
    TRANSACTION_COST = 0.001

    def __init__(self, initial_balance: float = 10_000.0, max_shares: int = 10):
        self.initial = initial_balance
        self.balance = initial_balance
        self.shares  = 0
        self.max_shares = max_shares
        self.n_trades   = 0

    def execute(self, action: int, price: float) -> None:
        if action == 1 and self.balance >= price and self.shares < self.max_shares:
            qty = min(int(self.balance // price), self.max_shares - self.shares)
            cost = qty * price * self.TRANSACTION_COST
            self.shares  += qty
            self.balance -= qty * price + cost
            self.n_trades += 1
        elif action == 2 and self.shares > 0:
            proceeds = self.shares * price
            self.balance += proceeds - proceeds * self.TRANSACTION_COST
            self.shares = 0
            self.n_trades += 1

    def value(self, price: float) -> float:
        return self.balance + self.shares * price


# ─── Chargement des modèles ──────────────────────────────────────────────────

def load_models() -> dict:
    """Charge LSTM + scaler + PPO pour chaque ticker disposant des 3 artefacts."""
    import tensorflow as tf
    from stable_baselines3 import PPO

    tf.get_logger().setLevel("ERROR")
    models = {}
    for fname in os.listdir(MODELS_DIR):
        if not (fname.startswith("lstm_") and fname.endswith(".keras")):
            continue
        k = fname[len("lstm_"):-len(".keras")]
        paths = {
            "lstm":   os.path.join(MODELS_DIR, f"lstm_{k}.keras"),
            "scaler": os.path.join(MODELS_DIR, f"scaler_{k}.pkl"),
            "ppo":    os.path.join(MODELS_DIR, f"ppo_{k}.zip"),
        }
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        with open(paths["scaler"], "rb") as f:
            scaler = pickle.load(f)
        models[k.replace("_", "-")] = {
            "lstm":   tf.keras.models.load_model(paths["lstm"], compile=False),
            "scaler": scaler,
            "ppo":    PPO.load(paths["ppo"]),
        }
        log.info(f"Modèles chargés : {k}")
    if not models:
        raise RuntimeError("Aucun modèle entraîné dans models/ — lance train.py d'abord")
    return models


def ppo_action_proba(agent, obs: np.ndarray) -> tuple[int, float]:
    """Action déterministe + probabilité réelle de la policy PPO."""
    import torch
    obs_t, _ = agent.policy.obs_to_tensor(obs)
    with torch.no_grad():
        probs = agent.policy.get_distribution(obs_t).distribution.probs.cpu().numpy()[0]
    action = int(np.argmax(probs))
    return action, float(probs[action])


def build_obs(row: dict, pred_price: float, trader: PaperTrader) -> np.ndarray:
    """Réplique exactement l'observation de env.TradingEnv._obs (12 valeurs)."""
    price = float(row["Close"]) + 1e-10
    value = trader.value(price)
    return np.array([
        pred_price / price,
        float(row.get("RSI", 50))        / 100.0,
        float(row.get("MACD", 0))        / price,
        float(row.get("EMA_20", price))  / price,
        float(row.get("EMA_50", price))  / price,
        float(row.get("BB_width", 0.05)),
        float(row.get("ATR", 0))         / price,
        float(row.get("Vol_ratio", 1.0)),
        float(row.get("Return", 0.0)),
        trader.shares  / trader.max_shares,
        trader.balance / trader.initial,
        (value / trader.initial) - 1.0,
    ], dtype=np.float32)


def connect_kafka():
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import NoBrokersAvailable

    for attempt in range(30):
        try:
            consumer = KafkaConsumer(
                TOPIC_IN,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=GROUP_ID,
                auto_offset_reset="earliest",
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            )
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
            log.info(f"Connecté à Kafka ({KAFKA_BOOTSTRAP})")
            return consumer, producer
        except NoBrokersAvailable:
            log.info(f"Kafka indisponible, retry {attempt + 1}/30 dans 5s...")
            time.sleep(5)
    raise RuntimeError(f"Impossible de joindre Kafka sur {KAFKA_BOOTSTRAP}")


def main():
    models = load_models()
    consumer, producer = connect_kafka()

    windows = {t: deque(maxlen=SEQ_LEN) for t in models}   # fenêtres de features
    traders = {t: PaperTrader()         for t in models}   # portefeuilles virtuels

    log.info(f"En écoute sur `{TOPIC_IN}` → signaux vers `{TOPIC_OUT}`")
    for msg in consumer:
        tick = msg.value
        ticker = tick.get("ticker")
        if ticker not in models:
            continue

        windows[ticker].append(tick)
        if len(windows[ticker]) < SEQ_LEN:
            continue            # préchauffage : pas encore 60 jours d'historique

        m, trader = models[ticker], traders[ticker]
        close = float(tick["Close"])

        # 1. LSTM : fenêtre de 60 jours → rendement J+1 prédit (%)
        raw = np.array([[float(r.get(f, 0.0)) for f in FEATURES]
                        for r in windows[ticker]])
        scaled = m["scaler"].transform(raw)
        r_hat  = float(m["lstm"].predict(
            scaled.reshape(1, SEQ_LEN, len(FEATURES)), verbose=0)[0][0])
        pred_price = close * (1 + r_hat / TARGET_SCALE)

        # 2. PPO : observation → action + probabilité réelle
        obs = build_obs(tick, pred_price, trader)
        action, conf = ppo_action_proba(m["ppo"], obs)

        # 3. Paper trading : on exécute l'action
        trader.execute(action, close)
        value = trader.value(close)

        signal = {
            "ticker":               ticker,
            "date":                 tick["date"],
            "close":                round(close, 2),
            "predicted_price":      round(pred_price, 2),
            "predicted_return_pct": round(r_hat, 3),
            "action":               AMAP[action],
            "confidence":           round(conf, 3),
            "shares":               trader.shares,
            "balance":              round(trader.balance, 2),
            "portfolio_value":      round(value, 2),
            "total_return_pct":     round((value / trader.initial - 1) * 100, 2),
            "n_trades":             trader.n_trades,
            "ts":                   time.time(),
        }
        producer.send(TOPIC_OUT, key=ticker, value=signal)
        if not tick.get("warmup"):
            log.info(f"{ticker} {tick['date']}  close={close:<8.2f} "
                     f"{AMAP[action]:<4} (p={conf:.2f})  "
                     f"portefeuille={value:,.0f}$ ({signal['total_return_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
