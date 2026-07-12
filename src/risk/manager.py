from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from src.config import CONFIG


class RiskManager:
    def __init__(self, signal: Dict[str, Any], account_size: float = 10_000_000):
        self.signal = signal
        self.account_size = account_size
        self.risk_config = CONFIG["risk"]
        self.indicators = signal.get("indicators", {})
        self.price = signal.get("price", 0) or signal.get("entry", 0) or self.indicators.get("close", 0)

    @property
    def atr_value(self) -> float:
        return self.indicators.get("atr_5", 0) or self.indicators.get("atr_15", 0) or 0

    def compute_entry(self) -> float:
        direction = self.signal.get("direction", "long")
        if direction == "long":
            return self.signal.get("entry", self.price) or self.price
        return self.signal.get("entry", self.price) or self.price

    def compute_stop_loss(self) -> float:
        direction = self.signal.get("direction", "long")
        atr = self.atr_value
        atr_mult = self.risk_config["atr_stop_multiplier"]
        entry = self.compute_entry()
        if direction == "long":
            stop = entry - (atr * atr_mult)
        else:
            stop = entry + (atr * atr_mult)
        return round(stop, 2)

    def compute_targets(self) -> Dict[str, float]:
        direction = self.signal.get("direction", "long")
        entry = self.compute_entry()
        stop = self.compute_stop_loss()
        atr = self.atr_value or abs(entry - stop)

        if direction == "long":
            t1 = entry + (atr * 1.0)
            t2 = entry + (atr * 1.5)
            t3 = entry + (atr * 2.0)
        else:
            t1 = entry - (atr * 1.0)
            t2 = entry - (atr * 1.5)
            t3 = entry - (atr * 2.0)

        return {
            "target_1": round(t1, 2),
            "target_2": round(t2, 2),
            "target_3": round(t3, 2),
        }

    def compute_risk_reward(self) -> float:
        entry = self.compute_entry()
        stop = self.compute_stop_loss()
        targets = self.compute_targets()
        risk = abs(entry - stop)
        if risk <= 0:
            return 0
        reward = abs(targets["target_3"] - entry)
        return round(reward / risk, 2)

    def compute_position_size(self) -> Dict[str, float]:
        entry = self.compute_entry()
        stop = self.compute_stop_loss()
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return {"shares": 0, "capital_risk": 0, "position_value": 0}
        max_risk = self.account_size * (self.risk_config["max_risk_per_trade_pct"] / 100)
        shares = max_risk / risk_per_share
        position_value = shares * entry
        max_pos = self.risk_config["max_position_size_crore"] * 10_000_000
        if position_value > max_pos:
            shares = max_pos / entry
            position_value = shares * entry
        return {
            "shares": int(shares),
            "capital_risk": round(max_risk, 2),
            "position_value": round(position_value, 2),
        }

    def compute_probability(self) -> int:
        rr = self.compute_risk_reward()
        score = self.signal.get("confidence", 50)
        base_prob = min(95, score)
        if rr >= 3.0:
            return min(95, base_prob + 5)
        if rr >= 2.0:
            return min(90, base_prob)
        return min(80, base_prob - 5)

    def get_all(self) -> Dict[str, Any]:
        entry = self.compute_entry()
        stop = self.compute_stop_loss()
        targets = self.compute_targets()
        rr = self.compute_risk_reward()
        pos = self.compute_position_size()
        prob = self.compute_probability()
        return {
            "entry": entry,
            "stop_loss": stop,
            "targets": targets,
            "risk_reward": rr,
            "position": pos,
            "probability": prob,
            "quality": self._get_quality(rr, prob),
        }

    def _get_quality(self, rr: float, prob: int) -> str:
        if rr >= 3.0 and prob >= 85: return "A+"
        if rr >= 2.5 and prob >= 80: return "A"
        if rr >= 2.0 and prob >= 70: return "B"
        return "C"