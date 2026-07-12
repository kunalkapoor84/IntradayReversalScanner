import numpy as np
import pandas as pd
import sys
sys.path.insert(0, r"C:\IntradayReversalScan")

from src.indicators.indicators import rsi, adx, atr, macd, vwap, ema, sma, _wilder_smooth

# Create sample data with known directional movement
np.random.seed(42)
n = 200
close = 100 + np.cumsum(np.random.randn(n) * 0.5)
high = close + np.abs(np.random.randn(n) * 0.3)
low = close - np.abs(np.random.randn(n) * 0.3)
volume = np.random.randint(100000, 1000000, n)

df = pd.DataFrame({
    "open": close - np.random.randn(n) * 0.2,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
})

print("=== VERIFICATION ===")
print(f"Bars: {len(df)}")
print()

# RSI
rsi_vals = rsi(df["close"], 14)
print(f"RSI(14) last 5: {[f'{v:.1f}' for v in rsi_vals.tail(5).values]}")
print(f"RSI(14) last: {rsi_vals.iloc[-1]:.2f}")

# ATR
atr_vals = atr(df, 14)
print(f"ATR(14) last: {atr_vals.iloc[-1]:.2f}")

# ADX
adx_vals = adx(df, 14)
print(f"ADX(14) last: {adx_vals.iloc[-1]:.2f}")
print(f"ADX(14) non-null count: {adx_vals.notna().sum()}")

# MACD
macd_data = macd(df["close"], 12, 26, 9)
print(f"MACD line last: {macd_data['macd'].iloc[-1]:.4f}")
print(f"MACD signal last: {macd_data['signal'].iloc[-1]:.4f}")
print(f"MACD hist last: {macd_data['histogram'].iloc[-1]:.4f}")

# VWAP
vwap_vals = vwap(df)
print(f"VWAP last: {vwap_vals.iloc[-1]:.2f}")

# Check if ADX is all NaN
print(f"ADX(14) all NaN: {adx_vals.isna().all()}")

# Check the intermediate values
up = df["high"].diff()
down = -(df["low"].diff())
print(f"up non-zero: {(up > 0).sum()}")
print(f"down non-zero: {(down > 0).sum()}")

# Check plus_dm and minus_dm
plus_dm = pd.Series(np.zeros(len(df)), index=df.index)
minus_dm = pd.Series(np.zeros(len(df)), index=df.index)
for i in range(1, len(df)):
    if up.iloc[i] > down.iloc[i] and up.iloc[i] > 0:
        plus_dm.iloc[i] = up.iloc[i]
    elif down.iloc[i] > up.iloc[i] and down.iloc[i] > 0:
        minus_dm.iloc[i] = down.iloc[i]
print(f"plus_dm non-zero: {(plus_dm > 0).sum()}")
print(f"minus_dm non-zero: {(minus_dm > 0).sum()}")

# Check _wilder_smooth
smoothed_plus = _wilder_smooth(plus_dm, 14)
print(f"smoothed_plus_dm non-null: {smoothed_plus.notna().sum()}")
print(f"smoothed_plus_dm first non-null index: {smoothed_plus.first_valid_index()}")
print(f"smoothed_plus_dm last: {smoothed_plus.iloc[-1]:.4f}")

# Check ATR
high, low, close = df["high"], df["low"], df["close"]
tr = pd.concat([
    high - low,
    (high - close.shift(1)).abs(),
    (low - close.shift(1)).abs(),
], axis=1).max(axis=1)
atr_vals = _wilder_smooth(tr, 14)
print(f"atr_vals non-null: {atr_vals.notna().sum()}")
print(f"atr_vals last: {atr_vals.iloc[-1]:.4f}")

# Check plus_di
plus_di = 100 * (smoothed_plus / atr_vals)
print(f"plus_di non-null: {plus_di.notna().sum()}")
print(f"plus_di last: {plus_di.iloc[-1]:.4f}")

# Check minus_di
smoothed_minus = _wilder_smooth(minus_dm, 14)
minus_di = 100 * (smoothed_minus / atr_vals)
print(f"minus_di non-null: {minus_di.notna().sum()}")
print(f"minus_di last: {minus_di.iloc[-1]:.4f}")

# Check denom
denom = (plus_di + minus_di).replace(0, np.nan)
print(f"denom non-null: {denom.notna().sum()}")
print(f"denom last: {denom.iloc[-1]:.4f}")

# Check dx
dx = 100 * ((plus_di - minus_di).abs() / denom)
print(f"dx non-null: {dx.notna().sum()}")
print(f"dx last: {dx.iloc[-1]:.4f}")

# Check ADX
adx_series = _wilder_smooth(dx, 14)
print(f"ADX non-null: {adx_series.notna().sum()}")
print(f"ADX last: {adx_series.iloc[-1]:.4f}")
print(f"ADX first valid index: {adx_series.first_valid_index()}")