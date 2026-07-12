from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def _ema_numba(values: np.ndarray, period: int) -> np.ndarray:
    result = np.empty_like(values)
    multiplier = 2.0 / (period + 1.0)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def ema(series: pd.Series, period: int) -> pd.Series:
    values = series.values.astype(np.float64)
    result = _ema_numba(values, period)
    return pd.Series(result, index=series.index)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    values = series.values.astype(np.float64)
    result = np.empty_like(values)
    result[:] = np.nan
    if len(values) < period:
        return pd.Series(result, index=series.index)
    first_valid = 0
    for start in range(len(values) - period + 1):
        window = values[start:start + period]
        if np.any(~np.isnan(window)):
            first_valid = start + period - 1
            with np.errstate(invalid='ignore'):
                result[first_valid] = np.nanmean(window)
            break
    else:
        return pd.Series(result, index=series.index)
    for i in range(first_valid + 1, len(values)):
        if np.isnan(values[i]):
            result[i] = result[i - 1]
        else:
            result[i] = (result[i - 1] * (period - 1) + values[i]) / period
    return pd.Series(result, index=series.index)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi.where(~(avg_gain.eq(0) & avg_loss.eq(0)), 50.0)


def vwap(df: pd.DataFrame, period: int = None) -> pd.Series:
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return pd.Series(np.nan, index=df.index)
    vwap_val = (df["close"] * df["volume"]).rolling(window=period or len(df)).sum() / df["volume"].rolling(window=period or len(df)).sum()
    return vwap_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return _wilder_smooth(true_range, period)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -(low.diff())
    plus_dm = pd.Series(np.zeros(len(df)), index=df.index)
    minus_dm = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(1, len(df)):
        if up.iloc[i] > down.iloc[i] and up.iloc[i] > 0:
            plus_dm.iloc[i] = up.iloc[i]
        elif down.iloc[i] > up.iloc[i] and down.iloc[i] > 0:
            minus_dm.iloc[i] = down.iloc[i]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_vals = _wilder_smooth(tr, period)
    atr_vals = atr_vals.replace(0, np.nan)
    plus_di = 100 * (_wilder_smooth(plus_dm, period) / atr_vals)
    minus_di = 100 * (_wilder_smooth(minus_dm, period) / atr_vals)
    di_sum = plus_di + minus_di
    dx = pd.Series(np.zeros(len(df)), index=df.index)
    mask = di_sum > 0
    dx[mask] = 100 * ((plus_di[mask] - minus_di[mask]).abs() / di_sum[mask])
    adx_series = _wilder_smooth(dx, period)
    return adx_series


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr_values = atr(df, period)
    hl_avg = (df["high"] + df["low"]) / 2
    upper_band = hl_avg + multiplier * atr_values
    lower_band = hl_avg - multiplier * atr_values
    st_series = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            prev = st_series.iloc[i - 1]
            prev_val = prev if not (isinstance(prev, float) and np.isnan(prev)) else lower_band.iloc[i]
            st_series.iloc[i] = max(lower_band.iloc[i], prev_val)
        else:
            prev = st_series.iloc[i - 1]
            prev_val = prev if not (isinstance(prev, float) and np.isnan(prev)) else upper_band.iloc[i]
            st_series.iloc[i] = min(upper_band.iloc[i], prev_val)

    return {"supertrend": st_series, "direction": direction}


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    low_k = df["low"].rolling(window=k_period).min()
    high_k = df["high"].rolling(window=k_period).max()
    k = 100 * ((df["close"] - low_k) / (high_k - low_k).replace(0, np.nan))
    d = k.rolling(window=d_period).mean()
    return {"k": k, "d": d}


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mean = tp.rolling(window=period).mean()
    mad = (tp - mean).abs().rolling(window=period).mean()
    mad = mad.replace(0, np.nan)
    return (tp - mean) / (0.015 * mad)


def obv(df: pd.DataFrame) -> pd.Series:
    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv[i] = obv[i - 1] + df["volume"].iloc[i]
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv[i] = obv[i - 1] - df["volume"].iloc[i]
        else:
            obv[i] = obv[i - 1]
    return pd.Series(obv, index=df.index)


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    mfv = df["volume"] * ((2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan))
    mfv_sum = mfv.rolling(window=period).sum()
    vol_sum = df["volume"].rolling(window=period).sum()
    return mfv_sum / vol_sum.replace(0, np.nan)


class IndicatorEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.computed: Dict[str, pd.Series] = {}

    def compute_all(self) -> Dict[str, pd.Series]:
        self.computed["ema_9"] = ema(self.df["close"], 9)
        self.computed["ema_20"] = ema(self.df["close"], 20)
        self.computed["ema_50"] = ema(self.df["close"], 50)
        self.computed["ema_200"] = ema(self.df["close"], 200)
        self.computed["vwap"] = vwap(self.df)
        self.computed["atr"] = atr(self.df)
        self.computed["rsi"] = rsi(self.df["close"])
        macd_data = macd(self.df["close"])
        self.computed["macd"] = macd_data["macd"]
        self.computed["macd_signal"] = macd_data["signal"]
        self.computed["macd_hist"] = macd_data["histogram"]
        self.computed["adx"] = adx(self.df)
        st = supertrend(self.df)
        self.computed["supertrend"] = st["supertrend"]
        self.computed["supertrend_dir"] = st["direction"]
        stoch = stochastic(self.df)
        self.computed["stoch_k"] = stoch["k"]
        self.computed["stoch_d"] = stoch["d"]
        self.computed["cci"] = cci(self.df)
        self.computed["obv"] = obv(self.df)
        self.computed["cmf"] = cmf(self.df)
        return self.computed

    def get_current(self) -> Dict[str, float]:
        result = {}
        for name, series in self.computed.items():
            if isinstance(series, pd.Series) and not series.empty:
                val = series.iloc[-1]
                result[name] = float(val) if not (isinstance(val, float) and np.isnan(val)) else 0.0
        return result