from __future__ import annotations

import asyncio
import gc
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from flask import Flask, jsonify, render_template, request

from src.config import CONFIG
from src.data.dhan_client import MarketDataManager
from src.indicators.indicators import IndicatorEngine, rsi, adx, atr, macd, vwap
from src.indicators.volume import VolumeAnalyzer
from src.risk.manager import RiskManager
from src.scanners.manager import ScannerManager
from src.scanners.scanners import (
    BearishPullbackScanner,
    BullishPullbackScanner,
    ExhaustionReversalScanner,
    FailedBreakdownScanner,
    FailedBreakoutScanner,
    TrendReversalScanner,
    VWAPReversalScanner,
)


def convert_val(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: convert_val(v) for k, v in v.items()}
    if isinstance(v, (list, tuple)):
        return [convert_val(x) for x in v]
    return v


app = Flask(__name__)
market_data = MarketDataManager()

@app.after_request
def cleanup(response):
    market_data.invalidate_cache()
    gc.collect()
    return response


def run_async(coro):
    return asyncio.run(coro)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan_single", methods=["POST"])
def scan_single():
    ticker = request.json.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        security_id = run_async(market_data.resolve_ticker(ticker))
        if security_id is None:
            return jsonify({"error": f"Ticker '{ticker}' not found"}), 404

        tf_data = run_async(market_data.get_multi_timeframe_data(security_id, CONFIG["timeframes"]))
        if not tf_data or all(df.empty for df in tf_data.values()):
            return jsonify({"error": f"No data for {ticker}"}), 404

        result = {
            "ticker": ticker,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "timeframes": {},
            "scanner_results": [],
        }

        for tf_name, df in tf_data.items():
            if df.empty or len(df) < 50:
                result["timeframes"][tf_name] = {"error": "insufficient data"}
                continue

            close = float(df["close"].iloc[-1])
            high = float(df["high"].iloc[-1])
            low = float(df["low"].iloc[-1])
            vol = int(df["volume"].iloc[-1])

            engine = IndicatorEngine(df)
            engine.compute_all()
            current = engine.get_current()

            vol_analyzer = VolumeAnalyzer(df)
            vol_analysis = vol_analyzer.get_analysis()

            tf_result = {
                "close": round(close, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "volume": vol,
                "bars": len(df),
                "rsi": round(current.get("rsi", 0), 2),
                "adx": round(current.get("adx", 0), 2),
                "atr": round(current.get("atr", 0), 4),
                "macd": round(current.get("macd", 0), 4),
                "macd_signal": round(current.get("macd_signal", 0), 4),
                "macd_hist": round(current.get("macd_hist", 0), 4),
                "vwap": round(current.get("vwap", 0), 2),
                "ema_9": round(current.get("ema_9", 0), 2),
                "ema_20": round(current.get("ema_20", 0), 2),
                "ema_50": round(current.get("ema_50", 0), 2),
                "ema_200": round(current.get("ema_200", 0), 2),
                "volume_ratio": vol_analysis.get("volume_ratio", 0),
                "avg_volume": vol_analysis.get("avg_volume", 0),
                "is_spike": vol_analysis.get("is_spike", False),
                "is_climax": vol_analysis.get("is_climax", False),
            }
            result["timeframes"][tf_name] = tf_result

        enabled_scanners = [
            s for s, c in CONFIG["scanners"].items()
            if isinstance(c, dict) and c.get("enabled", False)
        ]
        for stype in enabled_scanners:
            try:
                scanners_map = {
                    "bullish_pullback": BullishPullbackScanner,
                    "bearish_pullback": BearishPullbackScanner,
                    "exhaustion_reversal": ExhaustionReversalScanner,
                    "trend_reversal": TrendReversalScanner,
                    "vwap_reversal": VWAPReversalScanner,
                    "failed_breakdown": FailedBreakdownScanner,
                    "failed_breakout": FailedBreakoutScanner,
                }
                cls = scanners_map.get(stype)
                if cls is None:
                    continue
                scanner = cls(ticker, tf_data)
                signal = scanner.scan()
                if signal:
                    risk = RiskManager(signal)
                    rm = risk.get_all()
                    signal["risk_management"] = rm
                    result["scanner_results"].append({
                        "ticker": ticker,
                        "scanner": stype,
                        "signal": signal.get("signal", ""),
                        "direction": signal.get("direction", ""),
                        "confidence": signal.get("confidence", 0),
                        "entry": rm.get("entry", 0),
                        "stop_loss": rm.get("stop_loss", 0),
                        "risk_reward": rm.get("risk_reward", 0),
                        "quality": rm.get("quality", ""),
                        "targets": rm.get("targets", {}),
                    })
            except Exception as e:
                result["scanner_results"].append({
                    "scanner": stype,
                    "error": str(e)[:100],
                })

        return jsonify(convert_val(result))

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/scan_all", methods=["POST"])
def scan_all():
    try:
        df = run_async(market_data.build_universe())
        if df.empty:
            return jsonify({"error": "No universe data"}), 500

        tickers = df["security_id"].tolist() if "security_id" in df.columns else df.index.tolist()
        tickers = tickers[:CONFIG.get("scan_batch_size", 20)]

        manager = ScannerManager(market_data)
        signals = run_async(manager.scan_all(tickers))

        results = []
        for s in signals:
            try:
                risk = RiskManager(s)
                rm = risk.get_all()
                s["risk_management"] = rm
            except Exception:
                rm = {}
            results.append({
                "ticker": s.get("ticker", ""),
                "signal": s.get("signal", ""),
                "direction": s.get("direction", ""),
                "confidence": s.get("confidence", 0),
                "entry": rm.get("entry", 0),
                "stop_loss": rm.get("stop_loss", 0),
                "risk_reward": rm.get("risk_reward", 0),
                "quality": rm.get("quality", ""),
                "price": s.get("price", 0),
                "indicators": {
                    "rsi_5": s.get("indicators", {}).get("rsi_5", 0),
                    "adx_5": s.get("indicators", {}).get("adx_5", 0),
                    "atr_5": s.get("indicators", {}).get("atr_5", 0),
                },
            })

        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return jsonify(convert_val({
            "count": len(results),
            "signals": results[:20],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }))

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/search", methods=["POST"])
def search_tickers():
    query = request.json.get("query", "").strip().upper()
    if not query or len(query) < 1:
        return jsonify({"results": []})
    try:
        universe = run_async(market_data.build_universe())
        if universe.empty or "trading_symbol" not in universe.columns:
            return jsonify({"results": []})
        matches = universe[universe["trading_symbol"].str.contains(query, na=False)]
        results = []
        for _, row in matches.head(20).iterrows():
            results.append({
                "ticker": row.get("trading_symbol", ""),
                "name": row.get("name", ""),
                "security_id": row.get("security_id", ""),
            })
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quote", methods=["POST"])
def get_quote():
    ticker = request.json.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        quote = run_async(market_data.client.get_quote(ticker))
        return jsonify({"ticker": ticker, "quote": quote})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"Starting webapp on http://0.0.0.0:{port}")
    print("Open this URL in your phone browser to use the scanner")
    app.run(host="0.0.0.0", port=port, debug=False)