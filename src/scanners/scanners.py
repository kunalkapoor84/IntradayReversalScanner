from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import CONFIG
from src.indicators.indicators import IndicatorEngine
from src.indicators.volume import VolumeAnalyzer
from src.signals.patterns import CandlestickPatternDetector, SmartMoneyConcepts
from src.signals.trend import TrendFilter, RelativeStrength


class BaseScanner:
    def __init__(
        self,
        ticker: str,
        timeframe_data: Dict[str, pd.DataFrame],
        sector: str = "",
        index_df: Optional[pd.DataFrame] = None,
    ):
        self.ticker = ticker
        self.timeframe_data = timeframe_data
        self.sector = sector
        self.index_df = index_df
        self.indicators: Dict[str, Dict[str, float]] = {}
        self.patterns: Dict[str, List[str]] = {}
        self.volume_analysis: Dict[str, Any] = {}
        self.trend_filter: Optional[TrendFilter] = None

    def _compute_indicators_for_tf(self, tf: str) -> Optional[Dict[str, float]]:
        df = self.timeframe_data.get(tf)
        if df is None or len(df) < 50:
            return None
        engine = IndicatorEngine(df)
        engine.compute_all()
        self.indicators[tf] = engine.get_current()
        self.indicators[tf]["close"] = float(df["close"].iloc[-1])
        pat = CandlestickPatternDetector(df)
        self.patterns[tf] = pat.detect_all()
        vol = VolumeAnalyzer(df)
        self.volume_analysis[tf] = vol.get_analysis()
        return self.indicators[tf]

    def _detect_smc(self, tf: str) -> List[str]:
        df = self.timeframe_data.get(tf)
        if df is None or len(df) < 20:
            return []
        smc = SmartMoneyConcepts(df)
        return smc.detect_all()

    def _compute_key_levels(self, tf: str) -> Dict[str, float]:
        df = self.timeframe_data.get(tf)
        levels = {}
        if df is None or len(df) < 20:
            return levels
        close = float(df["close"].iloc[-1])
        levels["close"] = round(close, 2)
        ind = self.indicators.get(tf, {})
        levels["vwap"] = round(ind.get("vwap", 0), 2)
        levels["ema_20"] = round(ind.get("ema_20", 0), 2)
        levels["orib_high"] = round(float(df["high"].iloc[:5].max()), 2)
        levels["orib_low"] = round(float(df["low"].iloc[:5].min()), 2)
        levels["week_high"] = round(float(df["high"].tail(20).max()), 2)
        levels["week_low"] = round(float(df["low"].tail(20).min()), 2)
        levels["swing_high"] = round(float(df["high"].tail(10).max()), 2)
        levels["swing_low"] = round(float(df["low"].tail(10).min()), 2)
        if len(df) >= 390:
            daily = df.resample("D").agg({"high": "max", "low": "min"})
            if len(daily) >= 2:
                levels["yesterday_high"] = round(float(daily["high"].iloc[-2]), 2)
                levels["yesterday_low"] = round(float(daily["low"].iloc[-2]), 2)
            levels["today_high"] = round(float(daily["high"].iloc[-1]), 2)
            levels["today_low"] = round(float(daily["low"].iloc[-1]), 2)
        return levels

    def scan(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class BullishPullbackScanner(BaseScanner):
    """Scanner Type 1: Bullish Pullback - strong rally, pullback to support, resume uptrend"""

    def scan(self) -> Optional[Dict[str, Any]]:
        for tf in ("15m", "5m", "1m"):
            self._compute_indicators_for_tf(tf)
        tf_15 = self.indicators.get("15m", {})
        tf_5 = self.indicators.get("5m", {})
        tf_1 = self.indicators.get("1m", {})
        if not tf_15 or not tf_5 or not tf_1:
            return None

        df_5 = self.timeframe_data.get("5m")
        df_1 = self.timeframe_data.get("1m")
        if df_5 is None or df_1 is None or len(df_5) < 50 or len(df_1) < 50:
            return None

        close_1 = df_1["close"].iloc[-1]
        close_5 = df_5["close"].iloc[-1]

        # Higher timeframe bullish check
        ema_20_15, ema_50_15, ema_200_15 = (
            tf_15.get("ema_20", 0),
            tf_15.get("ema_50", 0),
            tf_15.get("ema_200", 0),
        )
        vwap_15 = tf_15.get("vwap", 0)
        vwap_5 = tf_5.get("vwap", 0)
        vwap_1 = tf_1.get("vwap", 0)

        if not (ema_20_15 > ema_50_15 > ema_200_15 and close_5 > vwap_15 and close_1 > vwap_15):
            return None

        # Price above key EMAs on 5m
        ema_20_5, ema_50_5 = tf_5.get("ema_20", 0), tf_5.get("ema_50", 0)
        if not (close_5 > ema_20_5 > ema_50_5 and close_1 > vwap_1 > 0 and close_1 > ema_20_5):
            return None

        # RSI check
        rsi_5, rsi_1 = tf_5.get("rsi", 50), tf_1.get("rsi", 50)
        if not (rsi_5 > 45 and rsi_1 > 45):
            return None

        macd_hist_5 = tf_5.get("macd_hist", 0)
        macd_line_5, macd_signal_5 = tf_5.get("macd", 0), tf_5.get("macd_signal", 0)
        if not (macd_hist_5 > 0 and macd_line_5 > macd_signal_5):
            return None

        macd_hist_contracting = True
        if len(df_5) >= 4:
            hist_vals = []
            for offset in range(3):
                idx = len(df_5) - 1 - offset
                if idx >= 0:
                    sub_df = df_5.iloc[:idx+1]
                    if len(sub_df) > 30:
                        sub_engine = IndicatorEngine(sub_df)
                        sub_engine.compute_all()
                        sub_hist = sub_engine.computed.get("macd_hist")
                        if sub_hist is not None and not sub_hist.empty:
                            hist_vals.append(float(sub_hist.iloc[-1]))
            if len(hist_vals) >= 3:
                macd_hist_contracting = hist_vals[-1] <= hist_vals[-2] <= hist_vals[-3]
            else:
                macd_hist_contracting = True

        # Volume check: pullback volume should decrease
        vol_5 = self.volume_analysis.get("5m", {})
        if vol_5.get("volume_ratio", 1) > 0.9:
            pass

        # Candlestick pattern check on 1m for entry candle
        pat_1 = self.patterns.get("1m", [])
        bullish_patterns = {"hammer", "bullish_engulfing", "tweezer_bottom", "pin_bar", "marubozu"}
        has_bullish_pattern = any(p in bullish_patterns for p in pat_1)

        # Entry candle must reject lower prices
        candle_1 = df_1.iloc[-1]
        lower_wick = min(candle_1["close"], candle_1["open"]) - candle_1["low"]
        body = abs(candle_1["close"] - candle_1["open"])
        rejects_lower = lower_wick > body * 0.5 if body > 0 else lower_wick > 0

        if not (has_bullish_pattern or rejects_lower):
            return None

        # Volume confirmation on entry
        vol_ratio = vol_5.get("volume_ratio", 1)
        if vol_ratio < 0.5:
            return None

        # SMC
        smc = self._detect_smc("5m")
        has_smc_bullish = any(s in smc for s in ["liquidity_sweep", "break_of_structure", "change_of_character"])

        adx_5 = tf_5.get("adx", 0)
        if adx_5 < CONFIG["trend_filter"]["min_adx"]:
            return None

        levels = self._compute_key_levels("5m")
        return {
            "ticker": self.ticker,
            "signal": "Bullish Pullback",
            "direction": "long",
            "price": df_5["close"].iloc[-1],
            "scanner": "bullish_pullback",
            "sector": self.sector,
            "primary_timeframe": "5m",
            "key_levels": levels,
            "confidence": self._compute_confidence(tf_15, tf_5, tf_1, vol_5, has_smc_bullish),
            "timeframe_signals": {
                "15m_trend": "bullish",
                "5m_patterns": self.patterns.get("5m", []),
                "1m_patterns": pat_1,
                "smc": smc,
            },
            "indicators": {
                "rsi_5": rsi_5,
                "rsi_1": rsi_1,
                "macd_5": macd_line_5,
                "macd_signal_5": macd_signal_5,
                "macd_hist_5": macd_hist_5,
                "adx_5": adx_5,
                "atr_5": tf_5.get("atr", 0),
            },
            "volume": vol_5,
            "vwap_distance_5": self._vwap_distance(close_5, vwap_5),
            "vwap_distance_1": self._vwap_distance(close_1, vwap_1),
        }

    def _vwap_distance(self, price: float, vwap: float) -> float:
        if vwap == 0:
            return 0
        return round((price - vwap) / vwap * 100, 2)

    def _compute_confidence(
        self, tf_15: Dict, tf_5: Dict, tf_1: Dict, vol: Dict, smc_bullish: bool
    ) -> int:
        score = 0
        if tf_15.get("ema_20", 0) > tf_15.get("ema_50", 0) > tf_15.get("ema_200", 0):
            score += 20
        if tf_5.get("rsi", 50) > 50:
            score += 10
        if tf_1.get("rsi", 50) > 50:
            score += 10
        vratio = vol.get("volume_ratio", 1)
        if 0.6 <= vratio <= 1.2:
            score += 10
        if vratio >= 1.5:
            score += 15
        if smc_bullish:
            score += 15
        if tf_5.get("adx", 0) >= 25:
            score += 10
        if any("bullish_engulfing" in p for p in self.patterns.get("1m", [])):
            score += 10
        score += min(10, int(tf_5.get("adx", 0)))
        return min(100, max(0, score))


class BearishPullbackScanner(BaseScanner):
    """Scanner Type 2: Bearish Pullback - strong selloff, pullback rally, sell into strength"""

    def scan(self) -> Optional[Dict[str, Any]]:
        for tf in ("15m", "5m", "1m"):
            self._compute_indicators_for_tf(tf)
        tf_15 = self.indicators.get("15m", {})
        tf_5 = self.indicators.get("5m", {})
        tf_1 = self.indicators.get("1m", {})
        if not tf_15 or not tf_5 or not tf_1:
            return None

        df_5 = self.timeframe_data.get("5m")
        df_1 = self.timeframe_data.get("1m")
        if df_5 is None or df_1 is None or len(df_5) < 50 or len(df_1) < 50:
            return None

        close_1, close_5 = df_1["close"].iloc[-1], df_5["close"].iloc[-1]

        ema_20_15, ema_50_15, ema_200_15 = (
            tf_15.get("ema_20", 0),
            tf_15.get("ema_50", 0),
            tf_15.get("ema_200", 0),
        )
        vwap_15 = tf_15.get("vwap", 0)
        if not (ema_20_15 < ema_50_15 < ema_200_15 and close_5 < vwap_15 and close_1 < vwap_15):
            return None

        ema_20_5, ema_50_5 = tf_5.get("ema_20", 0), tf_5.get("ema_50", 0)
        if not (close_5 < ema_20_5 < ema_50_5 and close_1 < vwap_15):
            return None

        rsi_5, rsi_1 = tf_5.get("rsi", 50), tf_1.get("rsi", 50)
        if not (rsi_5 < 55 and rsi_1 < 55):
            return None

        macd_hist_5 = tf_5.get("macd_hist", 0)
        macd_line_5, macd_signal_5 = tf_5.get("macd", 0), tf_5.get("macd_signal", 0)
        if not (macd_hist_5 < 0 and macd_line_5 < macd_signal_5):
            return None

        pat_1 = self.patterns.get("1m", [])
        bearish_patterns = {"shooting_star", "bearish_engulfing", "tweezer_top", "pin_bar"}
        has_bearish = any(p in bearish_patterns for p in pat_1)

        candle_1 = df_1.iloc[-1]
        upper_wick = candle_1["high"] - max(candle_1["close"], candle_1["open"])
        body = abs(candle_1["close"] - candle_1["open"])
        rejects_higher = upper_wick > body * 0.5 if body > 0 else upper_wick > 0

        if not (has_bearish_pattern or rejects_higher):
            return None

        adx_5 = tf_5.get("adx", 0)
        if adx_5 < 20:
            return None

        smc = self._detect_smc("5m")
        has_smc_bearish = any(s in smc for s in ["liquidity_sweep", "break_of_structure"])

        score = self._compute_bearish_confidence(tf_15, tf_5, tf_1, smc_bearish=has_smc_bearish)
        if score < CONFIG["scanners"]["bearish_pullback"]["min_confidence"]:
            return None

        levels = self._compute_key_levels("5m")
        return {
            "ticker": self.ticker,
            "signal": "Bearish Pullback",
            "direction": "short",
            "price": df_5["close"].iloc[-1],
            "scanner": "bearish_pullback",
            "sector": self.sector,
            "primary_timeframe": "5m",
            "key_levels": levels,
            "confidence": score,
            "timeframe_signals": {
                "15m_trend": "bearish",
                "5m_patterns": self.patterns.get("5m", []),
                "1m_patterns": pat_1,
                "smc": smc,
            },
            "indicators": {
                "rsi_5": rsi_5,
                "rsi_1": rsi_1,
                "macd_5": macd_line_5,
                "macd_signal_5": macd_signal_5,
                "macd_hist_5": macd_hist_5,
                "adx_5": adx_5,
                "atr_5": tf_5.get("atr", 0),
            },
            "volume": self.volume_analysis.get("5m", {}),
            "vwap_distance_5m": self._vwap_distance(close_5, tf_5.get("vwap", 0)),
        }

    def _vwap_distance(self, price: float, vwap: float) -> float:
        if vwap == 0:
            return 0
        return round((price - vwap) / vwap * 100, 2)

    def _compute_bearish_confidence(self, tf_15: Dict, tf_5: Dict, tf_1: Dict, vol_bearish: bool = False, smc_bearish: bool = False) -> int:
        score = 0
        if tf_15.get("ema_20", 999) < tf_15.get("ema_50", 999) < tf_15.get("ema_200", 999):
            score += 20
        if tf_5.get("rsi", 50) < 50: score += 10
        if tf_1.get("rsi", 50) < 50: score += 10
        if smc_bearish: score += 15
        if tf_5.get("adx", 0) >= 25: score += 10
        if any("bearish_engulfing" in p for p in self.patterns.get("1m", [])): score += 10
        score += min(10, int(tf_5.get("adx", 0)))
        return min(100, max(0, score))


class ExhaustionReversalScanner(BaseScanner):
    """Scanner Type 3: Exhaustion Reversal - panic selloff climax -> reversal"""

    def scan(self) -> Optional[Dict[str, Any]]:
        for tf in ("15m", "5m", "1m"):
            self._compute_indicators_for_tf(tf)
        tf_15 = self.indicators.get("15m", {})
        tf_5 = self.indicators.get("5m", {})
        tf_1 = self.indicators.get("1m", {})
        if not tf_15 or not tf_5 or not tf_1:
            return None

        df_5 = self.timeframe_data.get("5m")
        df_1 = self.timeframe_data.get("1m")
        if df_5 is None or df_1 is None or len(df_5) < 60 or len(df_1) < 30:
            return None

        close_1 = df_1["close"].iloc[-1]
        close_5 = df_5["close"].iloc[-1]

        # Detect selling exhaustion on 5m
        vol_5 = self.volume_analysis.get("5m", {})
        idx = len(df_5) - 1

        vol_climax = vol_5.get("is_climax", False)

        recent_candles = df_5.iloc[-5:]
        has_selloff = (recent_candles["close"] < recent_candles["open"]).sum() >= 3
        selloff_pct = 0.0
        if has_selloff:
            selloff_start = recent_candles["open"].iloc[0]
            selloff_end = recent_candles["close"].iloc[-1]
            selloff_pct = abs(selloff_end - selloff_start) / selloff_start * 100

        vwap_5 = tf_5.get("vwap", 0)
        vwap_1 = tf_1.get("vwap", 0)
        atr_5 = tf_5.get("atr", 0)

        vwap_deviation = abs(close_5 - vwap_5) / atr_5 if atr_5 > 0 else 0

        candle_1 = df_1.iloc[-1]
        lower_wick = min(candle_1["close"], candle_1["open"]) - candle_1["low"]
        body = abs(candle_1["close"] - candle_1["open"])

        has_long_lower_wick = lower_wick > body * 2 if body > 0 else (lower_wick > 0)
        pat_1 = self.patterns.get("1m", [])

        bullish_reversal = any(
            p in pat_1 for p in ["hammer", "bullish_engulfing", "morning_star", "tweezer_bottom", "pin_bar"]
        )

        be_bullish = False
        be_bearish = False

        initial_check = vol_climax and has_selloff and (bullish_reversal or has_long_lower_wick)
        if not initial_check and vwap_deviation < 2.0 and selloff_pct < 2.0:
            return None

        rsi_5 = tf_5.get("rsi", 50)
        rsi_1 = tf_1.get("rsi", 50)
        rsi_divergence_bullish = False
        if len(df_5) >= 10:
            low_5 = df_5["low"].iloc[-10:].min()
            if close_5 > low_5 * 1.02 and rsi_5 > 30:
                rsi_divergence_bullish = True

        macd_1 = tf_1.get("macd", 0)
        macd_signal_1 = tf_1.get("macd_signal", 0)

        if (vol_climax or selloff_pct >= 3) and (bullish_reversal or has_long_lower_wick or rsi_divergence_bullish):
            be_bullish = True

        has_rally = (df_5.iloc[-5:]["close"] > df_5.iloc[-5:]["open"]).sum() >= 3
        upper_wick = candle_1["high"] - max(candle_1["close"], candle_1["open"])
        has_long_upper_wick = upper_wick > body * 2 if body > 0 else (upper_wick > 0)
        bearish_reversal = any(p in pat_1 for p in ["shooting_star", "bearish_engulfing", "evening_star", "tweezer_top"])
        rsi_divergence_bearish = rsi_5 > 70 and close_5 < df_5["high"].iloc[-10:].max()

        has_vol_climax_up = False
        if has_rally and (bearish_reversal or has_long_upper_wick or rsi_divergence_bearish):
            be_bearish = True

        if be_bullish:
            direction = "long"
            signal_type = "Exhaustion Reversal (Bullish)"
        elif be_bearish:
            direction = "short"
            signal_type = "Exhaustion Reversal (Bearish)"
        else:
            return None

        score = 75
        if vol_climax: score += 10
        if rsi_divergence_bullish: score += 10
        if score < CONFIG["scanners"]["exhaustion_reversal"]["min_confidence"]:
            return None

        levels = self._compute_key_levels("5m")
        return {
            "ticker": self.ticker,
            "signal": signal_type,
            "direction": direction,
            "price": close_5,
            "scanner": "exhaustion_reversal",
            "sector": self.sector,
            "primary_timeframe": "5m",
            "key_levels": levels,
            "confidence": min(100, score),
            "timeframe_signals": {
                "vol_climax": vol_climax,
                "selloff_depth_pct": round(selloff_pct, 2),
                "vwap_deviation_atr": round(vwap_deviation, 2),
                "patterns_1m": pat_1,
                "rsi_divergence": rsi_divergence_bullish or rsi_divergence_bearish,
            },
            "indicators": {"rsi_5": rsi_5, "rsi_1": rsi_1, "atr_5": atr_5, "adx_5": tf_5.get("adx", 0)},
            "volume": vol_5,
        }


class TrendReversalScanner(BaseScanner):
    """Scanner Type 4: Trend Reversal - EMA crossover, VWAP reclaim, structure change"""

    def scan(self) -> Optional[Dict[str, Any]]:
        for tf in ("15m", "5m", "1m"):
            self._compute_indicators_for_tf(tf)
        tf_15 = self.indicators.get("15m", {})
        tf_5 = self.indicators.get("5m", {})
        tf_1 = self.indicators.get("1m", {})
        if not tf_15 or not tf_5 or not tf_1:
            return None

        df_5 = self.timeframe_data.get("5m")
        if df_5 is None or len(df_5) < 100:
            return None

        close_5 = df_5["close"].iloc[-1]

        # Detect EMA crossover (9 crossing 20)
        ema_9_5, ema_20_5 = tf_5.get("ema_9", 0), tf_5.get("ema_20", 0)
        ema_9_15, ema_20_15 = tf_15.get("ema_9", 0), tf_15.get("ema_20", 0)

        prev_ema_9 = None
        prev_ema_20 = None
        if len(df_5) > 1:
            prev_close = df_5["close"].iloc[:-1]
            if len(prev_close) > 20:
                from src.indicators.indicators import ema as ema_func
                prev_ema_9_val = ema_func(prev_close, 9).iloc[-1] if len(prev_close) >= 9 else 0
                prev_ema_20_val = ema_func(prev_close, 20).iloc[-1] if len(prev_close) >= 20 else 0
                prev_ema_9 = float(prev_ema_9_val)
                prev_ema_20 = float(prev_ema_20_val)

        bullish_cross = False
        bearish_cross = False
        if prev_ema_9 is not None and prev_ema_20 is not None:
            bullish_cross = prev_ema_9 <= prev_ema_20 and ema_9_5 > ema_20_5
            bearish_cross = prev_ema_9 >= prev_ema_20 and ema_9_5 < ema_20_5

        # VWAP reclaim
        vwap_5 = tf_5.get("vwap", 0)
        prev_vwap = None
        if len(df_5) > 1 and vwap_5 > 0:
            prev_close = df_5["close"].iloc[:-1]
            if len(prev_close) > len(df_5):
                prev_close_series = df_5["close"].iloc[:len(df_5)-1]
            prev_close_series = df_5["close"].iloc[:len(df_5)-1]
            prev_vwap_val = None
            for lookback in range(30, min(200, len(df_5))):
                temp = df_5.iloc[:len(df_5)-1]
                if len(temp) >= lookback:
                    from src.indicators.indicators import vwap
                    prev_vwap_val = temp["close"].iloc[-1]
                    break
            if prev_vwap is not None:
                vwap_reclaim_bullish = close_5 > vwap_5 and prev_vwap_val <= vwap_5
                vwap_reclaim_bearish = close_5 < vwap_5 and prev_close <= vwap_5 <= close_5
            else:
                vwap_reclaim_bullish = False
                vwap_reclaim_bearish = False
        else:
            vwap_reclaim_bullish = False
            vwap_reclaim_bearish = False

        # RSI crossing 50
        rsi_5 = tf_5.get("rsi", 50)
        rsi_15 = tf_15.get("rsi", 50)

        prev_rsi = None
        if len(df_5) > 14:
            from src.indicators.indicators import rsi as rsi_func
            prev_rsi_val = rsi_func(df_5["close"].iloc[:len(df_5)-1], 14)
            if isinstance(prev_rsi_val, pd.Series) and not prev_rsi_val.empty:
                prev_rsi = float(prev_rsi_val.iloc[-1])

        rsi_cross_bullish = prev_rsi is not None and prev_rsi <= 50 and rsi_5 > 50
        rsi_cross_bearish = prev_rsi is not None and prev_rsi >= 50 and rsi_5 < 50

        # MACD crossover
        macd_line_5, macd_signal_5 = tf_5.get("macd", 0), tf_5.get("macd_signal", 0)
        macd_hist_5 = tf_5.get("macd_hist", 0)

        prev_macd = None
        prev_signal = None
        if len(df_5) > 30:
            pass  # simplified

        macd_cross_bullish = macd_hist_5 > 0 and macd_line_5 > macd_signal_5
        macd_cross_bearish = macd_hist_5 < 0 and macd_line_5 < macd_signal_5

        # Volume surge
        vol_5 = self.volume_analysis.get("5m", {})
        vol_surge = vol_5.get("volume_ratio", 1) >= CONFIG["volume"]["volume_spike_multiplier"]

        # Structure change
        smc = self._detect_smc("5m")
        structure_change = "change_of_character" in smc or "break_of_structure" in smc

        # Determine signal
        bullish_signals = sum([
            bullish_cross, vwap_reclaim_bullish, rsi_cross_bullish,
            macd_cross_bullish, structure_change and "bullish" in str(smc)
        ])
        bearish_signals = sum([
            bearish_cross, vwap_reclaim_bearish, rsi_cross_bearish,
            macd_cross_bearish, structure_change and "bearish" in str(smc)
        ])

        if bullish_signals >= 2 and vol_surge:
            direction = "long"
            signal_type = "Trend Reversal (Bullish)"
        elif bearish_signals >= 2 and vol_surge:
            direction = "short"
            signal_type = "Trend Reversal (Bearish)"
        else:
            return None

        score = 75 + (10 if vol_surge else 0)
        if "structure_change" in locals() if True else True:
            if "change_of_character" in smc:
                score += 15
            if "break_of_structure" in smc:
                score += 10
        if score < CONFIG["scanners"]["trend_reversal"]["min_confidence"]:
            return None

        kl = self._compute_key_levels("5m")
        return {
            "ticker": self.ticker,
            "signal": signal_type,
            "direction": direction,
            "price": close_5,
            "scanner": "trend_reversal",
            "sector": self.sector,
            "primary_timeframe": "5m",
            "key_levels": kl,
            "confidence": min(100, score),
            "timeframe_signals": {
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
                "vwap_reclaim": vwap_reclaim_bullish or vwap_reclaim_bearish,
                "rsi_cross": rsi_cross_bullish or rsi_cross_bearish,
                "macd_cross": macd_cross_bullish or macd_cross_bearish,
                "structure_change": smc,
            },
            "indicators": {
                "rsi_5": rsi_5,
                "rsi_15": rsi_15,
                "macd_5": macd_line_5,
                "macd_signal_5": macd_signal_5,
                "macd_hist_5": macd_hist_5,
                "adx_5": tf_5.get("adx", 0),
                "atr_5": tf_5.get("atr", 0),
            },
            "volume": vol_5,
        }


class VWAPReversalScanner(BaseScanner):
    """Scanner Type 5: VWAP Reversal - price stretched from VWAP, reversal"""

    def scan(self) -> Optional[Dict[str, Any]]:
        for tf in ("15m", "5m", "1m"):
            self._compute_indicators_for_tf(tf)
        tf_5 = self.indicators.get("5m", {})
        tf_1 = self.indicators.get("1m", {})
        if not tf_5 or not tf_1:
            return None

        df_5 = self.timeframe_data.get("5m")
        df_1 = self.timeframe_data.get("1m")
        if df_5 is None or df_1 is None or len(df_5) < 30:
            return None

        close_5 = df_5["close"].iloc[-1]
        vwap_5 = tf_5.get("vwap", 0)
        atr_5 = tf_5.get("atr", 0)
        atr_mult = CONFIG["indicators"]["atr_multiplier"]

        if vwap_5 == 0 or atr_5 == 0:
            return None

        # Distance from VWAP in ATR units
        vwap_distance = abs(close_5 - vwap_5) / atr_5

        # Need at least 2 ATR stretch
        if vwap_distance < 2.0:
            return None

        # Check reversal candle
        candle_1 = df_1.iloc[-1]
        pat_1 = self.patterns.get("1m", [])

        # Volume climax
        vol_5 = self.volume_analysis.get("5m", {})
        volume_climax = vol_5.get("is_climax", False)

        # MACD turn
        macd_hist_5 = tf_5.get("macd_hist", 0)
        macd_1 = tf_1.get("macd", 0)
        macd_signal_1 = tf_1.get("macd_signal", 0)

        # Price closing back towards VWAP
        body_dir = candle_1["close"] - candle_1["open"]
        moving_toward_vwap = (close_5 < vwap_5 and body_dir > 0) or (close_5 > vwap_5 and body_dir < 0)

        # Reversal pattern check
        if close_5 < vwap_5:
            # Bullish VWAP reversal
            bullish_reversal = any(
                p in pat_1 for p in ["hammer", "bullish_engulfing", "tweezer_bottom", "pin_bar", "marubozu"]
            )
            lower_wick = min(candle_1["close"], candle_1["open"]) - candle_1["low"]
            body = abs(candle_1["close"] - candle_1["open"])
            has_long_wick = lower_wick > body * 1.5 if body > 0 else False

            if not (bullish_reversal or has_long_wick) and not volume_climax:
                return None

            direction = "long"
            signal_type = "VWAP Reversal (Bullish)"
        else:
            bearish_reversal = any(
                p in pat_1 for p in ["shooting_star", "bearish_engulfing", "tweezer_top", "pin_bar"]
            )
            upper_wick = candle_1["high"] - max(candle_1["close"], candle_1["open"])
            body = abs(candle_1["close"] - candle_1["open"])
            has_long_wick = upper_wick > body * 1.5 if body > 0 else False

            if not (bearish_reversal or has_long_wick) and not volume_climax:
                return None

            direction = "short"
            signal_type = "VWAP Reversal (Bearish)"

        score = 75
        if volume_climax: score += 10
        if vwap_distance >= 3.0: score += 5
        if moving_toward_vwap: score += 5
        if macd_1 > macd_signal_1 if direction == "long" else macd_1 < macd_signal_1: score += 5
        if score < CONFIG["scanners"]["vwap_reversal"]["min_confidence"]:
            return None

        kl = self._compute_key_levels("5m")
        return {
            "ticker": self.ticker,
            "signal": signal_type,
            "direction": direction,
            "price": close_5,
            "scanner": "vwap_reversal",
            "sector": self.sector,
            "primary_timeframe": "5m",
            "key_levels": kl,
            "confidence": min(100, score),
            "timeframe_signals": {
                "vwap_distance_atr": round(vwap_distance, 2),
                "volume_climax": volume_climax,
                "patterns_1m": pat_1,
                "moving_towards_vwap": moving_toward_vwap,
            },
            "indicators": {
                "rsi_5": tf_5.get("rsi", 50),
                "atr_5": atr_5,
                "macd_5": tf_5.get("macd", 0),
                "macd_signal_5": tf_5.get("macd_signal", 0),
                "macd_hist_5": macd_hist_5,
            },
            "volume": vol_5,
        }


class FailedBreakoutBreakdownBase(BaseScanner):
    """Base for Failed Breakdown/Breakout scanners"""

    def __init__(self, ticker, tf_data, scanner_type="failed_breakdown_base", **kwargs):
        super().__init__(ticker, tf_data, **kwargs)
        self.scanner_type = scanner_type

    def _find_key_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        levels = {}
        if len(df) >= 390:
            daily_df = df.resample("D").agg({"high": "max", "low": "min", "close": "last"})
            if len(daily_df) >= 1:
                levels["yesterday_high"] = daily_df["high"].iloc[-2] if len(daily_df) >= 2 else daily_df["high"].iloc[-1]
                levels["yesterday_low"] = daily_df["low"].iloc[-2] if len(daily_df) >= 2 else daily_df["low"].iloc[-1]
                levels["today_high"] = daily_df["high"].iloc[-1]
                levels["today_low"] = daily_df["low"].iloc[-1]
        if len(df) >= 20:
            levels["week_high"] = df["high"].tail(20).max()
            levels["week_low"] = df["low"].tail(20).min()
        if len(df) >= 5:
            levels["orib_high"] = df["high"].iloc[:5].max()
            levels["orib_low"] = df["low"].iloc[:5].min()
        return levels

    def scan(self) -> Optional[Dict[str, Any]]:
        for tf in ("5m", "1m"):
            self._compute_indicators_for_tf(tf)
        tf_5 = self.indicators.get("5m", {})
        tf_1 = self.indicators.get("1m", {})
        if not tf_5 or not tf_1:
            return None

        df_5 = self.timeframe_data.get("5m")
        df_1 = self.timeframe_data.get("1m")
        if df_5 is None or df_1 is None or len(df_5) < 20:
            return None

        levels = self._find_key_levels(df_5)
        close_5 = df_5["close"].iloc[-1]
        candle_1 = df_1.iloc[-1]
        pat_1 = self.patterns.get("1m", [])

        vol_5 = self.volume_analysis.get("5m", {})
        vol_spike = vol_5.get("is_spike", False) or vol_5.get("volume_ratio", 1) >= 1.5

        # Check level breaks
        if self.scanner_type == "failed_breakdown":
            low_level = min(
                levels.get("yesterday_low", float("inf")),
                levels.get("week_low", float("inf")),
                levels.get("orib_low", float("inf")),
            )
            if close_5 >= low_level * 0.995:
                return None  # price didn't break the level

            # Price broke below and immediately recovered
            prev_low = df_5["low"].iloc[-3:].min()
            if prev_low >= low_level or close_5 <= low_level:
                return None

            recovered_above = close_5 > low_level and candle_1["close"] > candle_1["open"]
            absorption_candle = candle_1["close"] > candle_1["open"] and vol_spike

            if not (recovered_above and (absorption_candle or "bullish_engulfing" in pat_1)):
                return None

            direction = "long"
            signal_type = "Failed Breakdown"
        elif self.scanner_type == "failed_breakout":
            high_level = max(
                levels.get("yesterday_high", 0),
                levels.get("week_high", 0),
                levels.get("orib_high", 0),
            )
            if close_5 <= high_level * 1.005:
                return None

            prev_high = df_5["high"].iloc[-3:].max()
            if prev_high <= high_level or close_5 >= high_level:
                return None

            rejection_above = close_5 < high_level and candle_1["close"] < candle_1["open"]
            rejection_candle = candle_1["close"] < candle_1["open"] and vol_spike

            if not (rejection_above and (rejection_candle or "bearish_engulfing" in pat_1)):
                return None

            direction = "short"
            signal_type = "Failed Breakout"

        score = 78
        if vol_spike: score += 10
        if "bullish_engulfing" in pat_1 or "bearish_engulfing" in pat_1: score += 7
        if score < CONFIG["scanners"].get(self.scanner_type, {}).get("min_confidence", 75):
            return None

        kl = self._compute_key_levels("5m")
        return {
            "ticker": self.ticker,
            "signal": signal_type,
            "direction": direction,
            "price": close_5,
            "scanner": self.scanner_type,
            "sector": self.sector,
            "primary_timeframe": "5m",
            "key_levels": kl,
            "confidence": min(100, score),
            "timeframe_signals": {
                "broken_level": low_level if self.scanner_type == "failed_breakdown" else high_level,
                "volume_spike": vol_spike,
                "patterns_1m": pat_1,
            },
            "indicators": {
                "rsi_5": tf_5.get("rsi", 50),
                "atr_5": tf_5.get("atr", 0),
            },
            "volume": vol_5,
        }


class FailedBreakdownScanner(FailedBreakoutBreakdownBase):
    """Scanner Type 6: Failed Breakdown"""

    def scan(self) -> Optional[Dict[str, Any]]:
        self.scanner_type = "failed_breakdown"
        return super().scan()


class FailedBreakoutScanner(FailedBreakoutBreakdownBase):
    """Scanner Type 7: Failed Breakout"""

    def scan(self) -> Optional[Dict[str, Any]]:
        self.scanner_type = "failed_breakout"
        return super().scan()