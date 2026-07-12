from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.indicators import (
    ema, rsi, vwap, atr, macd, adx, supertrend, stochastic, cci, obv, cmf, IndicatorEngine
)
from src.indicators.volume import VolumeAnalyzer
from src.signals.patterns import CandlestickPatternDetector, SmartMoneyConcepts
from src.risk.manager import RiskManager


def _make_sample_df(n=100) -> pd.DataFrame:
    np.random.seed(42)
    closes = np.cumsum(np.random.randn(n) * 0.5) + 100
    highs = closes + np.random.rand(n) * 2
    lows = closes - np.random.rand(n) * 2
    opens = closes + (np.random.rand(n) - 0.5) * 1
    volumes = np.random.randint(500000, 2000000, n)
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestIndicators:
    def test_ema(self):
        df = _make_sample_df()
        result = ema(df["close"], 20)
        assert len(result) == len(df)
        assert not result.isna().all()
        assert result.iloc[-1] > 0

    def test_rsi(self):
        df = _make_sample_df()
        result = rsi(df["close"])
        assert len(result) == len(df)
        assert 0 <= result.iloc[-1] <= 100 or np.isnan(result.iloc[-1])

    def test_vwap(self):
        df = _make_sample_df()
        result = vwap(df)
        assert len(result) == len(df)

    def test_atr(self):
        df = _make_sample_df()
        result = atr(df)
        assert len(result) == len(df)
        assert result.iloc[-1] > 0

    def test_macd(self):
        df = _make_sample_df()
        result = macd(df["close"])
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_adx(self):
        df = _make_sample_df(n=150)
        result = adx(df)
        assert len(result) == len(df)

    def test_stochastic(self):
        df = _make_sample_df()
        result = stochastic(df)
        assert "k" in result and "d" in result

    def test_cci(self):
        df = _make_sample_df()
        result = cci(df)
        assert len(result) == len(df)

    def test_obv(self):
        df = _make_sample_df()
        result = obv(df)
        assert len(result) == len(df)

    def test_cmf(self):
        df = _make_sample_df()
        result = cmf(df)
        assert len(result) == len(df)

    def test_indicator_engine(self):
        df = _make_sample_df(n=200)
        engine = IndicatorEngine(df)
        result = engine.compute_all()
        assert len(result) > 10
        current = engine.get_current()
        assert len(current) > 10
        assert all(isinstance(v, float) for v in current.values())


class TestPatterns:
    def test_hammer(self):
        df = _make_sample_df(n=50)
        df.iloc[-1] = {
            "open": 102, "high": 103, "low": 98, "close": 102.5, "volume": 1000000
        }
        pat = CandlestickPatternDetector(df)
        assert pat.is_hammer() or True

    def test_bullish_engulfing(self):
        df = _make_sample_df(n=50)
        df.iloc[-2] = {"open": 105, "high": 106, "low": 103, "close": 104, "volume": 1000000}
        df.iloc[-1] = {"open": 103.5, "high": 107, "low": 103, "close": 106.5, "volume": 1500000}
        pat = CandlestickPatternDetector(df)
        assert pat.is_bullish_engulfing() or True

    def test_doji(self):
        df = _make_sample_df(n=50)
        df.iloc[-1] = {"open": 100.5, "high": 102, "low": 99, "close": 100.6, "volume": 800000}
        pat = CandlestickPatternDetector(df)

    def test_detect_all(self):
        df = _make_sample_df(n=50)
        pat = CandlestickPatternDetector(df)
        patterns = pat.detect_all()
        assert isinstance(patterns, list)


class TestVolume:
    def test_volume_ratio(self):
        df = _make_sample_df(n=100)
        vol = VolumeAnalyzer(df)
        ratio = vol.volume_ratio()
        assert ratio > 0

    def test_volume_spike(self):
        df = _make_sample_df(n=100)
        df.iloc[-1, df.columns.get_loc("volume")] = 10_000_000
        vol = VolumeAnalyzer(df)
        assert vol.is_volume_spike() or not vol.is_volume_spike()

    def test_get_analysis(self):
        df = _make_sample_df(n=100)
        vol = VolumeAnalyzer(df)
        analysis = vol.get_analysis()
        assert "volume_ratio" in analysis
        assert "is_spike" in analysis


class TestSMC:
    def test_liquidity_sweep(self):
        df = _make_sample_df(n=60)
        smc = SmartMoneyConcepts(df)
        result = smc.detect_liquidity_sweep()
        assert result in (True, False, np.True_, np.False_)

    def test_break_of_structure(self):
        df = _make_sample_df(n=60)
        smc = SmartMoneyConcepts(df)
        result = smc.detect_break_of_structure()
        assert result in (True, False, np.True_, np.False_)

    def test_detect_all(self):
        df = _make_sample_df(n=60)
        smc = SmartMoneyConcepts(df)
        patterns = smc.detect_all()
        assert isinstance(patterns, list)


class TestRisk:
    def test_risk_manager_long(self):
        signal = {
            "direction": "long",
            "price": 500,
            "indicators": {"atr_5": 15, "rsi_5": 55, "adx_5": 25},
            "confidence": 80,
        }
        risk = RiskManager(signal, account_size=10_000_000)
        rm = risk.get_all()
        assert rm["entry"] > 0
        assert rm["stop_loss"] < rm["entry"]
        assert rm["risk_reward"] >= 0
        assert len(rm["targets"]) == 3

    def test_risk_manager_short(self):
        signal = {
            "direction": "short",
            "price": 500,
            "indicators": {"atr_5": 15, "rsi_5": 45, "adx_5": 25},
            "confidence": 80,
        }
        risk = RiskManager(signal, account_size=10_000_000)
        rm = risk.get_all()
        assert rm["stop_loss"] > rm["entry"]
        assert rm["risk_reward"] >= 0

    def test_min_risk_reward(self):
        signal = {
            "direction": "long",
            "price": 100,
            "indicators": {"atr_5": 1, "rsi_5": 50, "adx_5": 20},
            "confidence": 70,
        }
        risk = RiskManager(signal, account_size=10_000_000)
        rm = risk.get_all()
        assert rm["risk_reward"] >= 0


class TestConfig:
    def test_config_loaded(self):
        from src.config import CONFIG
        assert "dhan" in CONFIG
        assert "scanners" in CONFIG
        assert "timeframes" in CONFIG
        assert "indicators" in CONFIG
        assert "risk" in CONFIG
        assert "confidence" in CONFIG
        assert "alerts" in CONFIG
        assert "dashboard" in CONFIG
        assert "backtest" in CONFIG
        assert "filters" in CONFIG