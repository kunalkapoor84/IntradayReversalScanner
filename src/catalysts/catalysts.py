from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger


class CatalystDetector:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._last_fetch: Optional[datetime] = None

    async def detect(self, ticker: str) -> List[str]:
        catalysts = []

        earnings = await self._check_earnings(ticker)
        if earnings:
            catalysts.append(f"Earnings: {earnings}")

        news = await self._check_news(ticker)
        if news:
            catalysts.extend(news)

        return catalysts

    async def _check_earnings(self, ticker: str) -> Optional[str]:
        if ticker in self._cache and self._cache[ticker].get("earnings"):
            return self._cache[ticker]["earnings"]
        return None

    async def _check_news(self, ticker: str) -> List[str]:
        return []

    async def fetch_live_catalysts(self, tickers: List[str]) -> Dict[str, List[str]]:
        result = {}
        for t in tickers[:20]:
            result[t] = await self.detect(t)
            await asyncio.sleep(0.05)
        return result


class MarketContext:
    def __init__(self):
        self.nifty_data: Optional[Dict] = None
        self.banknifty_data: Optional[Dict] = None
        self.vix: float = 0
        self.advance_decline: Dict[str, int] = {"advances": 0, "declines": 0, "ratio": 0}
        self.sector_performance: Dict[str, float] = {}
        self.fii_dii: Dict[str, float] = {}
        self.lot_size: Dict[str, int] = {}
        self._cache: Dict = {}
        self._last_fetch: Optional[datetime] = None

    async def update(self):
        now = datetime.now()
        if self._last_fetch and (now - self._last_fetch).seconds < 60:
            return

        self._last_fetch = now
        logger.info("Updating market context...")
        await asyncio.sleep(0.1)

    def get_sector_trend(self, sector: str) -> str:
        perf = self.sector_performance.get(sector, 0)
        if perf > 0.5: return "bullish"
        if perf < -0.5: return "bearish"
        return "neutral"

    def get_summary(self) -> Dict[str, Any]:
        return {
            "nifty_change": self.nifty_data.get("change_pct", 0) if self.nifty_data else 0,
            "vix": self.vix,
            "advance_decline_ratio": self.advance_decline.get("ratio", 0),
            "market_bias": self._get_market_bias(),
            "updated": datetime.now().strftime("%H:%M:%S"),
        }

    def _get_market_bias(self) -> str:
        if self.nifty_data and self.nifty_data.get("change_pct", 0) > 0.5:
            return "bullish"
        if self.nifty_data and self.nifty_data.get("change_pct", 0) < -0.5:
            return "bearish"
        return "neutral"