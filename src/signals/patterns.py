from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class CandlestickPatternDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._compute_body_and_wicks()

    def _compute_body_and_wicks(self):
        df = self.df
        df["body"] = abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
        df["total_range"] = df["high"] - df["low"]
        df["body_pct"] = df["body"] / df["total_range"].replace(0, np.nan) * 100
        df["upper_wick_pct"] = df["upper_wick"] / df["total_range"].replace(0, np.nan) * 100
        df["lower_wick_pct"] = df["lower_wick"] / df["total_range"].replace(0, np.nan) * 100
        df["is_green"] = df["close"] > df["open"]
        df["is_red"] = df["close"] < df["open"]

    def is_hammer(self, i: int = -1) -> bool:
        df = self.df
        row = df.iloc[i]
        return (
            row["is_green"]
            and row["lower_wick"] >= 2 * row["body"]
            and row["upper_wick"] <= 0.3 * row["body"]
            and row["body_pct"] >= 5
            and row["body_pct"] <= 35
        )

    def is_shooting_star(self, i: int = -1) -> bool:
        df = self.df
        row = df.iloc[i]
        return (
            row["is_red"]
            and row["upper_wick"] >= 2 * row["body"]
            and row["lower_wick"] <= 0.3 * row["body"]
            and row["body_pct"] >= 5
            and row["body_pct"] <= 35
        )

    def is_bullish_engulfing(self, i: int = -1) -> bool:
        if len(self.df) < 2:
            return False
        df = self.df
        prev, curr = df.iloc[i - 1], df.iloc[i]
        return (
            prev["is_red"]
            and curr["is_green"]
            and curr["open"] <= prev["close"]
            and curr["close"] >= prev["open"]
            and curr["body"] > prev["body"]
        )

    def is_bearish_engulfing(self, i: int = -1) -> bool:
        if len(self.df) < 2:
            return False
        df = self.df
        prev, curr = df.iloc[i - 1], df.iloc[i]
        return (
            prev["is_green"]
            and curr["is_red"]
            and curr["open"] >= prev["close"]
            and curr["close"] <= prev["open"]
            and curr["body"] > prev["body"]
        )

    def is_inside_bar(self, i: int = -1) -> bool:
        if len(self.df) < 2:
            return False
        df = self.df
        prev, curr = df.iloc[i - 1], df.iloc[i]
        return curr["high"] <= prev["high"] and curr["low"] >= prev["low"]

    def is_outside_bar(self, i: int = -1) -> bool:
        if len(self.df) < 2:
            return False
        df = self.df
        prev, curr = df.iloc[i - 1], df.iloc[i]
        return curr["high"] > prev["high"] and curr["low"] < prev["low"]

    def is_doji(self, i: int = -1) -> bool:
        df = self.df
        row = df.iloc[i]
        return row["body_pct"] < 5 and row["total_range"] > 0

    def is_morning_star(self, i: int = -1) -> bool:
        if len(self.df) < 3:
            return False
        df = self.df
        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        return (
            c1["is_red"]
            and c2["body_pct"] < 10
            and c3["is_green"]
            and c3["close"] > (c1["open"] + c1["close"]) / 2
        )

    def is_evening_star(self, i: int = -1) -> bool:
        if len(self.df) < 3:
            return False
        df = self.df
        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        return (
            c1["is_green"]
            and c2["body_pct"] < 10
            and c3["is_red"]
            and c3["close"] < (c1["open"] + c1["close"]) / 2
        )

    def is_tweezer_bottom(self, i: int = -1) -> bool:
        if len(self.df) < 2:
            return False
        df = self.df
        prev, curr = df.iloc[i - 1], df.iloc[i]
        return (
            prev["is_red"]
            and curr["is_green"]
            and abs(prev["low"] - curr["low"]) / curr["total_range"] < 0.1
            and curr["close"] > prev["close"]
        )

    def is_tweezer_top(self, i: int = -1) -> bool:
        if len(self.df) < 2:
            return False
        df = self.df
        prev, curr = df.iloc[i - 1], df.iloc[i]
        return (
            prev["is_green"]
            and curr["is_red"]
            and abs(prev["high"] - curr["high"]) / curr["total_range"] < 0.1
            and curr["close"] < prev["close"]
        )

    def is_pin_bar(self, i: int = -1) -> bool:
        df = self.df
        row = df.iloc[i]
        return (
            (row["lower_wick"] >= 2 * row["body"] or row["upper_wick"] >= 2 * row["body"])
            and row["body_pct"] > 5
            and row["body_pct"] < 40
        )

    def is_marubozu(self, i: int = -1) -> bool:
        df = self.df
        row = df.iloc[i]
        return row["body_pct"] > 90 and row["upper_wick_pct"] < 3 and row["lower_wick_pct"] < 3

    def detect_all(self, i: int = -1) -> List[str]:
        patterns = []
        checks = [
            ("hammer", self.is_hammer),
            ("shooting_star", self.is_shooting_star),
            ("bullish_engulfing", self.is_bullish_engulfing),
            ("bearish_engulfing", self.is_bearish_engulfing),
            ("inside_bar", self.is_inside_bar),
            ("outside_bar", self.is_outside_bar),
            ("doji", self.is_doji),
            ("morning_star", self.is_morning_star),
            ("evening_star", self.is_evening_star),
            ("tweezer_bottom", self.is_tweezer_bottom),
            ("tweezer_top", self.is_tweezer_top),
            ("pin_bar", self.is_pin_bar),
            ("marubozu", self.is_marubozu),
        ]
        for name, func in checks:
            if func(i):
                patterns.append(name)
        return patterns

    def is_bullish_pattern(self, i: int = -1, threshold: int = 2) -> bool:
        bullish = {
            "hammer", "bullish_engulfing", "morning_star",
            "tweezer_bottom", "pin_bar", "marubozu",
        }
        patterns = self.detect_all(i)
        if not patterns:
            return False
        bullish_count = sum(1 for p in patterns if p in bullish)
        return bullish_count >= threshold or "bullish_engulfing" in patterns

    def is_bearish_pattern(self, i: int = -1, threshold: int = 2) -> bool:
        bearish = {
            "shooting_star", "bearish_engulfing", "evening_star",
            "tweezer_top", "pin_bar",
        }
        patterns = self.detect_all(i)
        if not patterns:
            return False
        bearish_count = sum(1 for p in patterns if p in bearish)
        return bearish_count >= threshold or "bearish_engulfing" in patterns


class SmartMoneyConcepts:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._compute_swings()

    def _compute_swings(self, lookback: int = 5):
        df = self.df
        df["swing_high"] = df["high"].rolling(window=lookback, center=True).max()
        df["swing_low"] = df["low"].rolling(window=lookback, center=True).min()
        df["is_swing_high"] = df["high"] == df["swing_high"]
        df["is_swing_low"] = df["low"] == df["swing_low"]
        df["higher_high"] = df["swing_high"] > df["swing_high"].shift(1)
        df["higher_low"] = df["swing_low"] > df["swing_low"].shift(1)
        df["lower_high"] = df["swing_high"] < df["swing_high"].shift(1)
        df["lower_low"] = df["swing_low"] < df["swing_low"].shift(1)

    def detect_liquidity_sweep(self, i: int = -1, lookback: int = 10) -> bool:
        df = self.df
        if i == -1:
            i = len(df) - 1
        recent_low = df["low"].iloc[i - lookback : i + 1].min()
        recent_high = df["high"].iloc[i - lookback : i + 1].max()
        curr = df.iloc[i]
        prev = df.iloc[i - 1] if i > 0 else curr
        is_green = curr["close"] > curr["open"]
        is_red = curr["close"] < curr["open"]
        sweep_low = (
            prev["low"] <= recent_low * 0.995
            and curr["close"] > prev["close"]
            and is_green
        )
        sweep_high = (
            prev["high"] >= recent_high * 1.005
            and curr["close"] < prev["close"]
            and is_red
        )
        return sweep_low or sweep_high

    def detect_break_of_structure(self, i: int = -1, lookback: int = 10) -> bool:
        df = self.df
        if i == -1:
            i = len(df) - 1
        if i < lookback + 2:
            return False
        prev_high = df["high"].iloc[i - lookback : i].max()
        prev_low = df["low"].iloc[i - lookback : i].min()
        curr = df.iloc[i]
        is_green = curr["close"] > curr["open"] if "is_green" not in df.columns else curr["is_green"]
        is_red = curr["close"] < curr["open"] if "is_red" not in df.columns else curr["is_red"]
        bos_bullish = curr["close"] > prev_high and is_green
        bos_bearish = curr["close"] < prev_low and is_red
        return bos_bullish or bos_bearish

    def detect_fvg(self, i: int = -1, gap_bars: int = 3) -> bool:
        df = self.df
        if i == -1:
            i = len(df) - 1
        if i < gap_bars:
            return False
        bar1 = df.iloc[i - 2]
        bar3 = df.iloc[i]
        fvg_bullish = bar3["low"] > bar1["high"]
        fvg_bearish = bar3["high"] < bar1["low"]
        return fvg_bullish or fvg_bearish

    def detect_change_of_character(self, i: int = -1, lookback: int = 15) -> bool:
        df = self.df
        if i == -1:
            i = len(df) - 1
        if i < lookback:
            return False
        recent = df.iloc[i - lookback : i + 1]
        was_bearish = recent["lower_high"].iloc[: -2].sum() > recent["higher_high"].iloc[: -2].sum()
        now_bullish = recent["higher_low"].iloc[-3:].sum() >= 2
        if was_bearish and now_bullish and (df.iloc[i]["close"] > df.iloc[i]["open"] if "is_green" not in df.columns else df.iloc[i]["is_green"]):
            return True
        was_bullish = recent["higher_low"].iloc[: -2].sum() > recent["lower_low"].iloc[: -2].sum()
        now_bearish = recent["lower_high"].iloc[-3:].sum() >= 2
        if was_bullish and now_bearish and (df.iloc[i]["close"] < df.iloc[i]["open"] if "is_red" not in df.columns else df.iloc[i]["is_red"]):
            return True
        return False

    def detect_all(self, i: int = -1) -> List[str]:
        smc = []
        if self.detect_liquidity_sweep(i):
            smc.append("liquidity_sweep")
        if self.detect_break_of_structure(i):
            smc.append("break_of_structure")
        if self.detect_fvg(i):
            smc.append("fair_value_gap")
        if choc := self.detect_change_of_character(i):
            smc.append("change_of_character")
            smc.append(f"direction={'bullish' if choc else 'bearish'}")
        return smc