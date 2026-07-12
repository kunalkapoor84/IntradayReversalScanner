from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import CONFIG
from src.indicators.indicators import IndicatorEngine


class TrendFilter:
    def __init__(self, timeframe_data: Dict[str, pd.DataFrame]):
        self.data = timeframe_data
        self.tf_config = CONFIG["trend_filter"]
        self.higher_tf = self.tf_config["higher_tf"]
        self.mid_tf = self.tf_config["mid_tf"]
        self.entry_tf = self.tf_config["entry_tf"]
        self.ema_alignment_required = self.tf_config["ema_alignment_required"]

    def _get_trend_for_timeframe(self, tf: str) -> Dict[str, Any]:
        df = self.timeframe.get(tf)
        if df is None or len(df) < 50:
            return {"trend": "neutral", "strength": 0, "reasons": ["insufficient_data"]}

        engine = IndicatorEngine(df)
        indicators = engine.compute_all()
        current = engine.get_current()

        reasons = []
        trend = "neutral"
        strength = 0

        ema_20 = current.get("ema_20", 0)
        ema_50 = current.get("ema_50", 0)
        ema_200 = current.get("ema_200", 0)
        close = df["close"].iloc[-1]
        vwap = current.get("vwap", 0)
        rsi_val = current.get("rsi", 50)
        adx_val = current.get("adx", 0)
        macd_hist = current.get("macd_hist", 0)
        macd_line = current.get("macd", 0)
        macd_signal = current.get("macd_signal", 0)

        is_bullish = True
        is_bearish = True
        bull_reasons = []
        bear_reasons = []

        if self.ema_alignment_required:
            if ema_20 > ema_50 > ema_200 and close > ema_20:
                bull_reasons.append("ema_bullish_alignment")
            else:
                is_bullish = False
                bull_reasons.append("ema_not_aligned")

            if ema_20 < ema_50 < ema_200 and close < ema_20:
                bear_reasons.append("ema_bearish_alignment")
                is_bullish = False
            else:
                is_bearish = False
                bear_reasons.append("ema_not_aligned")
        else:
            if close > ema_20 and ema_20 > ema_50:
                bull_reasons.append("price_above_ema20")

        if close > vwap and vwap > 0:
            bull_reasons.append("above_vwap")
        else:
            is_bullish = False
        if close < vwap and vwap > 0:
            bear_reasons.append("below_vwap")
        else:
            is_bearish = False

        if rsi_val > 50:
            bull_reasons.append(f"rsi_{rsi_val:.1f}")
        else:
            is_bullish = False
        if rsi_val < 50:
            bear_reasons.append(f"rsi_{rsi_val:.1f}")
        else:
            is_bearish = False

        if macd_hist > 0 and macd_line > macd_signal:
            bull_reasons.append("macd_bullish")
        else:
            is_bullish = False
        if macd_hist < 0 and macd_line < macd_signal:
            bear_reasons.append("macd_bearish")
        else:
            is_bearish = False

        if adx_val >= CONFIG["trend_filter"]["min_adx"]:
            strength = min(100, int(adx_val))
            if is_bullish:
                bull_reasons.append(f"strong_trend_adx_{adx_val:.1f}")
            if is_bearish:
                bear_reasons.append(f"strong_trend_adx_{adx_val:.1f}")
        else:
            strength = 0

        if is_bullish:
            trend = "bullish"
            strength = max(strength, 60)
            return {"trend": "bullish", "strength": strength, "reasons": bull_reasons}
        elif is_bearish:
            trend = "bearish"
            strength = max(strength, 60)
            return {"trend": "bearish", "strength": strength, "reasons": bear_reasons}
        else:
            return {"trend": "neutral", "strength": 15, "reasons": ["mixed_signals"]}

    def analyze(self) -> Dict[str, Any]:
        higher = self._get_trend_for_timeframe(self.higher_tf)
        mid = self._get_trend_for_timeframe(self.mid_tf)

        trend = "neutral"
        strength = 0
        reasons = []

        if higher["trend"] == "bullish" and mid["trend"] == "bullish":
            trend = "strongly_bullish"
            strength = min(100, higher["strength"] + 10)
            reasons = ["higher_tf_bullish", "mid_tf_bullish"]
        elif higher["trend"] == "bullish" and mid["trend"] == "neutral":
            trend = "bullish"
            strength = higher["strength"]
            reasons = ["higher_tf_bullish", "mid_tf_neutral"]
        elif higher["trend"] == "bearish" and mid["trend"] == "bearish":
            trend = "strongly_bearish"
            strength = min(100, higher["strength"] + 10)
            reasons = ["higher_tf_bearish", "mid_tf_bearish"]
        elif higher["trend"] == "bearish" and mid["trend"] == "neutral":
            trend = "bearish"
            strength = higher["strength"]
            reasons = ["higher_tf_bearish", "mid_tf_neutral"]
        elif higher["trend"] == "bullish" and mid["trend"] == "bearish":
            trend = "conflicted"
            strength = 30
            reasons = ["higher_tf_bullish_vs_mid_tf_bearish"]
        elif higher["trend"] == "bearish" and mid["trend"] == "bullish":
            trend = "conflicted"
            strength = 30
            reasons = ["higher_tf_bearish_vs_mid_tf_bullish"]
        else:
            trend = "neutral"
            strength = higher["strength"]
            reasons = ["no_clear_trend"]

        return {
            "higher_tf": higher,
            "mid_tf": mid,
            "trend": trend,
            "strength": strength,
            "reasons": reasons,
        }

    def is_bullish_biased(self) -> bool:
        return self.analyze()["trend"] in ("strongly_bullish", "bullish")

    def is_bearish_biased(self) -> bool:
        return self.analyze()["trend"] in ("strongly_bearish", "bearish")


class RelativeStrength:
    def __init__(self, stock_df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None):
        self.stock_df = stock_df
        self.index_df = index_df

    def stock_vs_nifty(self) -> float:
        if self.index_df is None or len(self.stock_df) < 20 or len(self.index_df) < 20:
            return 50.0
        stock_return = (
            self.stock_df["close"].iloc[-1] / self.stock_df["close"].iloc[-20] - 1
        ) * 100
        index_return = (
            self.index_df["close"].iloc[-1] / self.index_df["close"].iloc[-20] - 1
        ) * 100
        if abs(index_return) < 0.01:
            return 50.0
        return 50 + min(50, max(-50, stock_return - index_return))

    def score(self) -> int:
        rs = self.stock_vs_nifty()
        if rs >= 60:
            return 100
        if rs >= 50:
            return 75
        if rs >= 40:
            return 50
        return 25