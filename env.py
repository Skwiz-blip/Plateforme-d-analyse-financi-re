

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_OK = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_OK = True
    except ImportError:
        GYM_OK = False

if GYM_OK:
    _TradingEnvBase = gym.Env
else:
    _TradingEnvBase = object


class TradingEnv(_TradingEnvBase):
    metadata = {"render_modes": ["human"]}

    TRANSACTION_COST = 0.001

    def __init__(self, df: pd.DataFrame, lstm_preds: np.ndarray,
                 initial_balance: float = 10_000.0, max_shares: int = 10):
        super().__init__()
        self.df            = df.reset_index(drop=True)
        self.lstm_preds    = lstm_preds
        self.initial_bal   = initial_balance
        self.max_shares    = max_shares

        if GYM_OK:
            self.action_space      = spaces.Discrete(3)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            )
        self.reset()

    def reset(self, seed=None, options=None):
        if GYM_OK:
            super().reset(seed=seed)
        self.balance   = self.initial_bal
        self.shares    = 0
        self.idx       = 0
        self.history   = [self.initial_bal]
        self.trades    = []
        self.total_transaction_costs = 0.0
        return self._obs(), {}

    def step(self, action: int):
        if self.idx >= len(self.df) - 1:
            return self._obs(), 0.0, True, False, {}

        price     = float(self.df["Close"].iloc[self.idx])
        prev_val  = self.balance + self.shares * price

        if action == 1 and self.balance >= price and self.shares < self.max_shares:
            qty           = min(int(self.balance // price), self.max_shares - self.shares)
            cost          = qty * price * self.TRANSACTION_COST
            self.shares  += qty
            self.balance -= qty * price + cost
            self.total_transaction_costs += cost
            self.trades.append({"step": self.idx, "type": "BUY",
                                 "price": price, "qty": qty, "cost": cost})

        elif action == 2 and self.shares > 0:
            proceeds      = self.shares * price
            cost          = proceeds * self.TRANSACTION_COST
            self.balance += proceeds - cost
            self.total_transaction_costs += cost
            self.trades.append({"step": self.idx, "type": "SELL",
                                 "price": price, "qty": self.shares, "cost": cost})
            self.shares = 0

        self.idx += 1
        new_price = float(self.df["Close"].iloc[self.idx])
        new_val   = self.balance + self.shares * new_price
        self.history.append(new_val)

        reward = (new_val - prev_val) / (prev_val + 1e-10)
        if action == 0 and self.shares == 0:
            reward -= 0.0001

        done = self.idx >= len(self.df) - 1
        return self._obs(), float(reward), done, False, {}

    def _obs(self) -> np.ndarray:
        i     = min(self.idx, len(self.df) - 1)
        row   = self.df.iloc[i]
        price = float(row["Close"]) + 1e-10

        return np.array([
            float(self.lstm_preds[i]) / price,
            float(row.get("RSI", 50))                   / 100.0,
            float(row.get("MACD", 0))                   / price,
            float(row.get("EMA_20", price))             / price,
            float(row.get("EMA_50", price))             / price,
            float(row.get("BB_width", 0.05)),
            float(row.get("ATR", 0))                    / price,
            float(row.get("Vol_ratio", 1.0)),
            float(row.get("Return", 0.0)),
            self.shares    / self.max_shares,
            self.balance   / self.initial_bal,
            (self.history[-1] / self.initial_bal) - 1.0,
        ], dtype=np.float32)

    def metrics(self) -> dict:
        portfolio = np.array(self.history)
        returns   = np.diff(portfolio) / (portfolio[:-1] + 1e-10)

        total_return = (portfolio[-1] / self.initial_bal - 1) * 100
        sharpe = ((returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252)
                  if len(returns) > 1 else 0.0)

        running_max  = np.maximum.accumulate(portfolio)
        max_drawdown = ((portfolio - running_max) / (running_max + 1e-10)).min() * 100

        pairs = []
        buy_stack = []
        for t in self.trades:
            if t["type"] == "BUY":
                buy_stack.append(t)
            elif t["type"] == "SELL" and buy_stack:
                pairs.append((buy_stack.pop(0), t))
        wins = sum(1 for b, s in pairs if s["price"] > b["price"])
        win_rate = wins / len(pairs) * 100 if pairs else 0.0

        return {
            "total_return":  round(float(total_return), 2),
            "sharpe_ratio":  round(float(sharpe), 4),
            "max_drawdown":  round(float(max_drawdown), 2),
            "win_rate":      round(float(win_rate), 2),
            "n_trades":      len(self.trades),
            "final_balance": round(float(portfolio[-1]), 2),
        }
