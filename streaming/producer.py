import os
import json
import time
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("producer")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TICKERS         = [t.strip() for t in os.getenv("TICKERS", "AAPL").split(",") if t.strip()]
SPEED           = float(os.getenv("SPEED", "2.0"))
REPLAY_DAYS     = int(os.getenv("REPLAY_DAYS", "250"))
LOOP            = os.getenv("LOOP", "1") == "1"

DATA_DIR  = "data/processed"
TOPIC     = "prices"
SEQ_LEN   = 60          # fenêtre nécessaire au consumer avant son 1er signal
WARMUP_DELAY = 0.02     # les SEQ_LEN premiers ticks partent vite (préchauffage)


def connect_producer():
    """Connexion au broker avec retry — Kafka peut mettre ~30s à démarrer."""
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable

    for attempt in range(30):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
            log.info(f"Connecté à Kafka ({KAFKA_BOOTSTRAP})")
            return producer
        except NoBrokersAvailable:
            log.info(f"Kafka indisponible, retry {attempt + 1}/30 dans 5s...")
            time.sleep(5)
    raise RuntimeError(f"Impossible de joindre Kafka sur {KAFKA_BOOTSTRAP}")


def load_ticker(ticker: str) -> pd.DataFrame | None:
    """Charge le CSV processed et garde la fin : SEQ_LEN (préchauffage) + REPLAY_DAYS."""
    path = os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}.csv")
    if not os.path.exists(path):
        log.warning(f"{ticker} : {path} introuvable — ignoré")
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.tail(SEQ_LEN + REPLAY_DAYS)


def row_message(ticker: str, date, row: pd.Series, warmup: bool) -> dict:
    """Construit le message JSON d'un tick (toutes les colonnes numériques)."""
    msg = {"ticker": ticker, "date": date.strftime("%Y-%m-%d"), "warmup": warmup}
    for col, val in row.items():
        try:
            msg[col] = round(float(val), 6)
        except (TypeError, ValueError):
            pass
    return msg


def main():
    frames = {}
    for t in TICKERS:
        df = load_ticker(t)
        if df is not None:
            frames[t] = df
            log.info(f"{t} : {len(df)} jours à rejouer "
                     f"({df.index.min().date()} → {df.index.max().date()})")
    if not frames:
        raise RuntimeError("Aucune donnée à rejouer — lance d'abord train.py --step data")

    producer = connect_producer()
    max_len  = max(len(df) for df in frames.values())

    while True:
        log.info(f"Replay : {len(frames)} ticker(s), vitesse {SPEED} tick/s "
                 f"(préchauffage {SEQ_LEN} ticks accéléré)")
        sent = 0
        for i in range(max_len):
            warmup = i < SEQ_LEN
            for ticker, df in frames.items():
                if i >= len(df):
                    continue
                date, row = df.index[i], df.iloc[i]
                producer.send(TOPIC, key=ticker, value=row_message(ticker, date, row, warmup))
                sent += 1
            producer.flush()
            if not warmup and i % 50 == 0:
                log.info(f"tick {i}/{max_len} — {sent} messages envoyés")
            time.sleep(WARMUP_DELAY if warmup else 1.0 / SPEED)

        log.info(f"Replay terminé : {sent} messages")
        if not LOOP:
            break
        log.info("LOOP=1 → on rejoue depuis le début")

    producer.close()


if __name__ == "__main__":
    main()
