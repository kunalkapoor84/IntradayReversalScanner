from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from src.config import CONFIG


class VolumeAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vol_config = CONFIG["volume"]
        self.avg_period = self.vol_config["avg_period"]
        self.spike_ratio = self.vol_config["volume_spike_multiplier"]
        self.climax_ratio = self.vol_config["climax_ratio"]
        self.dry_up_ratio = self.vol_config["dry_up_ratio"]

    def avg_volume(self) -> float:
        if len(self.df) < self.avg_period:
            return float(self.df["volume"].mean())
        return float(self.df["volume"].tail(self.avg_period).mean())

    def volume_ratio(self) -> float:
        recent = self.df["volume"].iloc[-1]
        avg = self.avg_volume()
        if avg == 0:
            return 1.0
        return float(recent / avg)

    def is_volume_spike(self, i: int = -1) -> bool:
        return self.volume_ratio() >= self.spike_ratio

    def is_volume_climax(self, i: int = -1, lookback: int = 50) -> bool:
        df = self.df
        idx = i if i != -1 else len(df) - 1
        if idx < lookback:
            return False
        max_vol_in_window = df["volume"].iloc[idx - lookback : idx].max()
        if max_vol_in_window == 0:
            return False
        return df["volume"].iloc[idx] >= max_vol_in_window * self.climax_ratio

    def is_dry_up_volume(self, i: int = -1, lookback: int = 20) -> bool:
        df = self.df
        idx = i if i != -1 else len(df) - 1
        if idx < lookback:
            return False
        avg_vol = df["volume"].iloc[idx - lookback : idx].mean()
        if avg_vol == 0:
            return False
        return df["volume"].iloc[idx] <= avg_vol * self.dry_up_ratio

    def is_absorption(self, i: int = -1, lookback: int = 3) -> bool:
        df = self.df
        idx = i if i != -1 else len(df) - 1
        if idx < lookback:
            return False
        recent = df.iloc[idx - lookback + 1 : idx + 1]
        avg_range = recent["high"] - recent["low"]
        avg_vol = recent["volume"].mean()
        avg_vol = avg_vol if not np.isnan(avg_vol) else 0
        if avg_vol == 0:
            return False
        prev_avg_range = df.iloc[idx - lookback * 2 : idx - lookback + 1]["high"].mean() - df.iloc[
            idx - lookback * 2 : idx - lookback + 1
        ]["low"].mean()
        return avg_vol > self.spike_ratio * df["volume"].iloc[: idx - lookback + 1].mean() and avg_range < prev_avg_range * 0.5

    def is_stopping_volume(self, i: int = -1) -> bool:
        df = self.df
        idx = i if i != -1 else len(df) - 1
        if idx < 2:
            return False
        prev, curr = df.iloc[idx - 1], df.iloc[idx]
        vol_increase = curr["volume"] > prev["volume"] * 1.5
        price_narrow = abs(curr["close"] - curr["open"]) < (
            abs(prev["close"] - prev["open"]) * 0.5
            if abs(prev["close"] - prev["open"]) > 0
            else 0.5
        )
        long_wick = (
            curr["lower_wick"] > curr["body"] * 2
            if "lower_wick" in df.columns
            else False
        )
        return vol_increase and (price_narrow or long_wick)

    def effort_vs_result(self, i: int = -1) -> str:
        df = self.df
        idx = i if i != -1 else len(df) - 1
        if idx < self.avg_period:
            return "insufficient_data"
        avg_vol = df["volume"].iloc[idx - self.avg_period : idx].mean()
        avg_range = (
            df["high"].iloc[idx - self.avg_period : idx].mean()
            - df["low"].iloc[idx - self.avg_period : idx].mean()
        )
        if avg_vol == 0 or avg_range == 0:
            return "no_data"
        curr_vol = df["volume"].iloc[idx]
        curr_range = df["high"].iloc[idx] - df["low"].iloc[idx]
        vol_ratio = curr_vol / avg_vol
        range_ratio = curr_range / avg_range

        if vol_ratio > 1.5 and range_ratio < 0.7:
            return "absorption"
        if vol_ratio > 1.5 and range_ratio > 1.5:
            return "high_effort_high_result"
        if vol_ratio < 0.5 and range_ratio < 0.5:
            return "low_interest"
        return "neutral"

    def delivery_pct_increasing(self) -> bool:
        if "delivery_pct" not in self.df.columns:
            return False
        return self.df["delivery_pct"].tail(5).is_monotonic_increasing

    def above_20dma_volume(self) -> bool:
        if len(self.df) < 20:
            return False
        avg_20 = self.df["volume"].tail(20).mean()
        return self.df["volume"].iloc[-1] > avg_20

    def get_analysis(self) -> Dict:
        idx = len(self.df) - 1
        return {
            "volume_ratio": round(self.volume_ratio(), 2),
            "avg_volume": int(self.avg_volume()),
            "is_spike": self.is_volume_spike(),
            "is_climax": self.is_volume_climax(),
            "is_dry_up": self.is_dry_up_volume(),
            "is_absorption": self.is_absorption(),
            "is_stopping": self.is_stopping_volume(),
            "effort_result": self.effort_vs_result(),
            "above_20dma": self.above_20dma_volume(),
        }