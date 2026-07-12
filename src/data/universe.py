from __future__ import annotations

from typing import Any, Dict, List

from src.data.dhan_client import DhanHTTPClient, MarketDataManager


class UniverseBuilder:
    def __init__(self, market_data: MarketDataManager):
        self.market_data = market_data
        self.universe: Dict[str, Any] = {}

    async def build(self, force_refresh: bool = False) -> Dict[str, Any]:
        df = await self.market_data.build_universe(force_refresh)
        if df.empty:
            return {}
        self.universe = {
            "tickers": df["security_id"].tolist() if "security_id" in df.columns else df.index.tolist(),
            "dataframe": df,
            "count": len(df),
            "built_at": "now",
        }
        return self.universe

    def get_tickers(self) -> List[str]:
        return self.universe.get("tickers", [])

    def get_count(self) -> int:
        return self.universe.get("count", 0)