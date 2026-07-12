from __future__ import annotations

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

    async def scan_all(self, tickers: List[str]) -> List[Dict[str, Any]]:
        enabled_scanners = [
            stype for stype, cfg in self.scanner_config.items()
            if isinstance(cfg, dict) and cfg.get("enabled", False)
        ]
        self.all_results = []

        logger.info(f"Scanning {len(tickers)} tickers with {len(enabled_scanners)} scanners")

        for ticker in tickers[:50]:  # batch of 50 per cycle
            try:
                tf_data = await self.market_data.get_multi_timeframe_data(
                    ticker, CONFIG["timeframes"]
                )
                if not tf_data or all(df.empty for df in tf_data.values()):
                    continue

                results = self._scan_single(ticker, tf_data, enabled_scanners)
                self.all_results.extend(results)
            except Exception as e:
                logger.debug(f"Error scanning {ticker}: {e}")
                continue

        logger.info(f"Generated {len(self.all_results)} signals")
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