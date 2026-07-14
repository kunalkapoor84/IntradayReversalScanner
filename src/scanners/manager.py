from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.config import CONFIG
from src.data.universe import UniverseBuilder
from src.scanners.scanners import (
    BearishPullbackScanner,
    BullishPullbackScanner,
    ExhaustionReversalScanner,
    FailedBreakdownScanner,
    FailedBreakoutScanner,
    TrendReversalScanner,
    VWAPReversalScanner,
)


def ist_now() -> str:
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")


class ScannerManager:
    def __init__(self, market_data):
        self.market_data = market_data
        self.universe_builder = UniverseBuilder(market_data)
        self.scanner_config = CONFIG["scanners"]
        self.all_results: List[Dict[str, Any]] = []

    def _get_scanner(self, scanner_type: str, ticker: str, tf_data: Dict, **kwargs):
        scanners = {
            "bullish_pullback": BullishPullbackScanner,
            "bearish_pullback": BearishPullbackScanner,
            "exhaustion_reversal": ExhaustionReversalScanner,
            "trend_reversal": TrendReversalScanner,
            "vwap_reversal": VWAPReversalScanner,
            "failed_breakdown": FailedBreakdownScanner,
            "failed_breakout": FailedBreakoutScanner,
        }
        cls = scanners.get(scanner_type)
        if cls is None:
            return None
        return cls(ticker, tf_data, **kwargs)

    async def scan_all(
        self, security_ids: List[str], ticker_map: Dict[str, str] = None, limit: int = 0
    ) -> List[Dict[str, Any]]:
        enabled_scanners = [
            stype for stype, cfg in self.scanner_config.items()
            if isinstance(cfg, dict) and cfg.get("enabled", False)
        ]
        self.all_results = []
        ticker_map = ticker_map or {}

        if limit and len(security_ids) > limit:
            security_ids = security_ids[:limit]
            logger.info(f"{ist_now()} | Limited to {limit} tickers")

        semaphore = asyncio.Semaphore(1)
        logger.info(f"{ist_now()} | Scanning {len(security_ids)} tickers with {len(enabled_scanners)} scanners")

        async def _scan_one(sid: str) -> List[Dict[str, Any]]:
            ticker = ticker_map.get(sid, sid)
            async with semaphore:
                try:
                    bulk_tfs = [tf for tf in CONFIG["timeframes"] if tf["interval"] != "1m"]
                    tf_data = await self.market_data.get_multi_timeframe_data(
                        sid, bulk_tfs
                    )
                    if not tf_data or all(df.empty for df in tf_data.values()):
                        return []
                    results = self._scan_single(ticker, tf_data, enabled_scanners)
                    for r in results:
                        r["ticker"] = ticker
                    return results
                except Exception as e:
                    logger.debug(f"Error scanning {ticker}: {e}")
                    return []

        task_results = await asyncio.gather(
            *[_scan_one(sid) for sid in security_ids], return_exceptions=True
        )
        for res in task_results:
            if isinstance(res, list):
                self.all_results.extend(res)

        logger.info(f"{ist_now()} | Generated {len(self.all_results)} signals")
        return self.all_results

    def _scan_single(
        self, ticker: str, tf_data: Dict[str, pd.DataFrame], enabled_scanners: List[str]
    ) -> List[Dict[str, Any]]:
        results = []
        for stype in enabled_scanners:
            scanner_class = self._get_scanner(stype, ticker, tf_data)
            if scanner_class is None:
                continue
            try:
                result = scanner_class.scan()
                if result is not None:
                    result["ticker"] = ticker
                    if "5m" in scanner_class.indicators:
                        result["vwap_5"] = scanner_class.indicators["5m"].get("vwap", 0)
                        result["close_5"] = scanner_class.indicators["5m"].get("close", 0)
                    results.append(result)
            except Exception as e:
                logger.debug(f"Scanner {stype} error on {ticker}: {e}")
        return results

    def get_signals(self, min_score: int = 0) -> List[Dict[str, Any]]:
        scores = CONFIG.get("confidence", {})
        min_display = min_score or scores.get("min_display_score", 75)
        return [s for s in self.all_results if s.get("confidence", 0) >= min_display]

    def get_top_signals(self, n: int = 10) -> List[Dict[str, Any]]:
        return sorted(
            self.get_signals(), key=lambda x: x.get("confidence", 0), reverse=True
        )[:n]

    def clear_results(self):
        self.all_results.clear()