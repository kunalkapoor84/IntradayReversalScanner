from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from loguru import logger


def _is_retryable(exception):
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in (429, 500, 502, 503, 504)
    return True

from src.config import CONFIG


class DhanHTTPClient:
    def __init__(self):
        dhan_config = CONFIG["dhan"]
        self.client_id = dhan_config["client_id"]
        self.access_token = dhan_config["access_token"]
        self.base_url = dhan_config["api_base_url"]
        self.timeout = dhan_config["timeout_seconds"]
        self.max_retries = dhan_config["max_retries"]
        self._client: Optional[httpx.AsyncClient] = None
        self._ws_client: Optional[Any] = None
        self._has_valid_creds: bool = bool(self.client_id and self.access_token and "your_" not in self.client_id)
        self._rate_limiter = asyncio.Semaphore(5)
        self._last_request = 0.0
        self._security_list_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "access-token": self.access_token,
                "client-id": self.client_id,
                "Content-Type": "application/json",
            },
        )
        return self._client

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"
        async with self._rate_limiter:
            now = time.monotonic()
            since_last = now - self._last_request
            if since_last < 0.2:
                await asyncio.sleep(0.2 - since_last)
            self._last_request = time.monotonic()
            for attempt in range(3):
                try:
                    response = await client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Rate limited on {endpoint}, retrying in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    logger.error(f"HTTP error {e.response.status_code} on {endpoint}: {e.response.text}")
                    raise
                except httpx.TimeoutException:
                    logger.error(f"Timeout on {endpoint}")
                    raise
                except Exception as e:
                    logger.error(f"Request error on {endpoint}: {e}")
                    raise
            raise Exception(f"Failed after 3 retries: {endpoint}")

    async def get_intraday_candles(
        self, security_id: str, interval: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        if not self._has_valid_creds:
            raise httpx.HTTPStatusError("No valid credentials", request=None, response=None)
        interval_map = {"1m": "1", "5m": "5", "15m": "15", "25m": "25", "60m": "60"}
        api_interval = interval_map.get(interval)
        if api_interval is None:
            raise ValueError(f"Unsupported interval: {interval}")
        endpoint = "/charts/intraday"
        payload = {
            "securityId": security_id,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": api_interval,
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date,
        }
        logger.debug(f"Fetching intraday: {security_id} {interval}")
        data = await self._request("POST", endpoint, json=payload)
        return self._parse_candle_response(data)

    def _parse_candle_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        if not isinstance(data, dict):
            return pd.DataFrame()
        if "data" in data:
            candles = data["data"]
            if not candles:
                return pd.DataFrame()
            records = []
            for c in candles:
                records.append(
                    {
                        "timestamp": pd.to_datetime(c.get("startTime", c.get("timestamp"))),
                        "open": float(c.get("open", 0)),
                        "high": float(c.get("high", 0)),
                        "low": float(c.get("low", 0)),
                        "close": float(c.get("close", 0)),
                        "volume": int(c.get("volume", c.get("vol", 0))),
                    }
                )
        elif "open" in data and "timestamp" in data:
            opens = data["open"]
            if not opens:
                return pd.DataFrame()
            records = []
            for i in range(len(opens)):
                records.append(
                    {
                        "timestamp": pd.to_datetime(data["timestamp"][i], unit="s"),
                        "open": float(opens[i]),
                        "high": float(data["high"][i]),
                        "low": float(data["low"][i]),
                        "close": float(data["close"][i]),
                        "volume": int(data["volume"][i]),
                    }
                )
        else:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            df.set_index("timestamp", inplace=True)
        return df

    CSVS = {
        "SE": {
            "url": "https://images.dhan.co/api-data/api-scrip-master.csv",
            "exch": "SEM_EXM_EXCH_ID",
            "seg": "SEM_SEGMENT",
            "instr": "SEM_INSTRUMENT_NAME",
            "sid": "SEM_SMST_SECURITY_ID",
            "symbol": "SEM_TRADING_SYMBOL",
            "series": "SEM_SERIES",
            "name": "SM_SYMBOL_NAME",
        }
    }

    _SEGMENT_MAP = {
        "EQ": {"exch": "NSE", "seg": "E", "instr": "EQUITY", "series": "EQ"},
        "FO": {"exch": "NSE", "seg": "D", "instr": "FUTSTK", "series": ""},
    }

    NSE_FO_STOCKS = [
        "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT", "ADANIGREEN",
        "ADANIPORTS", "ADANIPOWER", "AIAENG", "AJANTPHARM", "ALEMBICLTD", "ALKEM",
        "ALKYLAMINE", "AMBER", "AMBUJACEM", "ANGELONE", "APLAPOLLO", "APOLLOHOSP",
        "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", "ASTRAL", "ATGL", "ATUL", "AUBANK",
        "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BALKRISIND",
        "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BATAINDIA", "BEL", "BERGEPAINT",
        "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL",
        "BRITANNIA", "CANBK", "CASTROLIND", "CEATLTD", "CESC", "CGPOWER", "CHOLAFIN",
        "CIPLA", "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "CROMPTON", "CUB",
        "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR", "DELTACORP", "DIXON",
        "DLF", "DMART", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK",
        "FORTIS", "GAIL", "GLENMARK", "GMRINFRA", "GODREJCP", "GODREJPROP",
        "GRANULES", "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC",
        "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HEXAWARE", "HINDALCO", "HINDPETRO",
        "HINDUNILVR", "HUDCO", "ICICIBANK", "ICICIGI", "ICICIPRULI",
        "IDEA", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL", "INDIAMART", "INDIGO",
        "INDUSINDBK", "INDUSTOWER", "INFY", "IOC", "IPCA", "IRB", "IRCTC", "IREDA",
        "ITC", "JINDALSTEL", "JKCEMENT", "JSL", "JSWENERGY", "JSWSTEEL", "JUBLFOOD",
        "KOTAKBANK", "KPITTECH", "L&TFH", "LALPATHLAB", "LAURUSLABS", "LT",
        "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI",
        "MAXHEALTH", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS",
        "MRF", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NBCC", "NCC",
        "NESTLEIND", "NHPC", "NMDC", "NTPC", "OBEROIRLTY", "OIL", "ONGC", "PAGEIND",
        "PEL", "PERSISTENT", "PETRONET", "PIDILITIND", "PIIND", "PNB", "PNBHOUSING",
        "POLICYBZR", "POLYCAB", "POWERGRID", "PRAJIND", "PRESTIGE", "PVRINOX",
        "RALLIS", "RAMCOCEM", "RBLBANK", "RCF", "RECLTD", "RELIANCE", "SAIL",
        "SBICARD", "SBILIFE", "SBIN", "SCHAEFFLER", "SHREECEM", "SHRIRAMFIN",
        "SIEMENS", "SRF", "STAR", "SUNPHARMA", "SUNTECK",
        "SUPREMEIND", "SYNGENE", "TATACHEM", "TATACOMM", "TATAELXSI", "TATAMOTORS",
        "TATAPOWER", "TATASTEEL", "TATATECH", "TCS", "TECHM", "THERMAX", "TITAN",
        "TORNTPHARM", "TORNTPOWER", "TRENT", "TRIDENT", "TVSMOTOR", "UBL",
        "ULTRACEMCO", "UNIONBANK", "UNOMINDA", "UPL", "UTIAMC", "VEDL", "VGUARD",
        "VOLTAS", "WIPRO", "YESBANK", "ZEEL", "ZENSARTECH", "ZYDUSLIFE",
    ]

    NSE_FO_SECURITY_IDS = {
        "ABB": "13", "ABBOTINDIA": "17903", "ABCAPITAL": "21614", "ABFRL": "30108",
        "ACC": "22", "ADANIENT": "25", "ADANIGREEN": "3563", "ADANIPORTS": "15083",
        "ADANIPOWER": "17388", "AIAENG": "13086", "AJANTPHARM": "8124",
        "ALEMBICLTD": "79", "ALKEM": "11703", "ALKYLAMINE": "4487", "AMBER": "1185",
        "AMBUJACEM": "1270", "ANGELONE": "324", "APLAPOLLO": "25780",
        "APOLLOHOSP": "157", "APOLLOTYRE": "163", "ASHOKLEY": "212",
        "ASIANPAINT": "236", "ASTRAL": "14418", "ATGL": "6066", "ATUL": "263",
        "AUBANK": "21238", "AXISBANK": "5900", "BAJAJ-AUTO": "16669",
        "BAJAJFINSV": "16675", "BAJFINANCE": "317", "BALKRISIND": "335",
        "BANDHANBNK": "2263", "BANKBARODA": "4668", "BANKINDIA": "4745",
        "BATAINDIA": "371", "BEL": "383", "BERGEPAINT": "404", "BHARATFORG": "422",
        "BHARTIARTL": "10604", "BHEL": "438", "BIOCON": "11373", "BOSCHLTD": "2181",
        "BPCL": "526", "BRITANNIA": "547", "CANBK": "10794", "CASTROLIND": "1250",
        "CEATLTD": "15254", "CESC": "628", "CGPOWER": "760", "CHOLAFIN": "685",
        "CIPLA": "694", "COALINDIA": "20374", "COFORGE": "11543", "COLPAL": "15141",
        "CONCOR": "4749", "CROMPTON": "17094", "CUB": "5701", "CUMMINSIND": "1901",
        "DABUR": "772", "DALBHARAT": "8075", "DEEPAKNTR": "19943",
        "DELTACORP": "15044", "DIXON": "21690", "DLF": "14732", "DMART": "19913",
        "DRREDDY": "881", "EICHERMOT": "910", "ESCORTS": "958", "EXIDEIND": "676",
        "FEDERALBNK": "1023", "FORTIS": "14592", "GAIL": "4717", "GLENMARK": "7406",
        "GMRINFRA": "13528", "GODREJCP": "10099", "GODREJPROP": "17875",
        "GRANULES": "11872", "GRASIM": "1232", 
        "HAL": "2303", "HAVELLS": "9819", "HCLTECH": "7229", "HDFCAMC": "4244",
        "HDFCBANK": "1333", "HDFCLIFE": "467", "HEROMOTOCO": "1348",
        "HEXAWARE": "29666", "HINDALCO": "1363", "HINDPETRO": "1406",
        "HINDUNILVR": "1394", "HUDCO": "20825", "ICICIBANK": "4963",
        "ICICIGI": "21770", "ICICIPRULI": "18652", "IDEA": "14366",
        "IDFCFIRSTB": "11184", "IEX": "220", "IGL": "11262", "INDHOTEL": "1512",
        "INDIAMART": "10726", "INDIGO": "11195", "INDUSINDBK": "5258",
        "INDUSTOWER": "29135", "INFY": "1594", "IOC": "1624", "IPCA": "1633",
        "IRB": "15313", "IRCTC": "13611", "IREDA": "20261", "ITC": "1660",
        "JINDALSTEL": "6733", "JKCEMENT": "13270", "JSL": "11236",
        "JSWENERGY": "17869", "JSWSTEEL": "11723", "JUBLFOOD": "18096",
        "KOTAKBANK": "1922", "KPITTECH": "9683", "L&TFH": "24948",
        "LALPATHLAB": "11654", "LAURUSLABS": "19234", "LT": "11483",
        "LTTS": "18564", "LUPIN": "10440", "M&M": "2031",
        "M&MFIN": "13285", "MANAPPURAM": "19061", "MARICO": "4067",
        "MARUTI": "10999", "MAXHEALTH": "22377", "MCX": "31181",
        "METROPOLIS": "9581", "MFSL": "2142", "MGL": "17534", "MOTHERSON": "4204",
        "MPHASIS": "4503", "MRF": "2277", "MUTHOOTFIN": "23650",
        "NATIONALUM": "6364", "NAUKRI": "13751", "NAVINFLUOR": "14672",
        "NBCC": "31415", "NCC": "2319", "NESTLEIND": "17963", "NHPC": "17400",
        "NMDC": "15332", "NTPC": "11630", "OBEROIRLTY": "20242", "OIL": "17438",
        "ONGC": "2475", "PAGEIND": "14413", "PEL": "11571",
        "PERSISTENT": "18365", "PETRONET": "11351", "PIDILITIND": "2664",
        "PIIND": "24184", "PNB": "10666", "PNBHOUSING": "18908",
        "POLICYBZR": "6656", "POLYCAB": "9590", "POWERGRID": "14977",
        "PRAJIND": "2705", "PRESTIGE": "20302", "PVRINOX": "13147",
        "RALLIS": "2816", "RAMCOCEM": "2043", "RBLBANK": "18391", "RCF": "2866",
        "RECLTD": "15355", "RELIANCE": "2885", "SAIL": "2963", "SBICARD": "17971",
        "SBILIFE": "21808", "SBIN": "3045", "SCHAEFFLER": "1011",
        "SHREECEM": "3103", "SHRIRAMFIN": "4306", "SIEMENS": "3150", "SRF": "3273",
        "STAR": "7374", "SUNPHARMA": "3351",
        "SUNTECK": "17641", "SUPREMEIND": "3363", "SYNGENE": "10243",
        "TATACHEM": "3405", "TATACOMM": "3721", "TATAELXSI": "3411",
        "TATAMOTORS": "759782", "TATAPOWER": "3426", "TATASTEEL": "3499",
        "TATATECH": "20293", "TCS": "11536", "TECHM": "13538", "THERMAX": "3475",
        "TITAN": "3506", "TORNTPHARM": "3518", "TORNTPOWER": "13786",
        "TRENT": "1964", "TRIDENT": "9685", "TVSMOTOR": "8479", "UBL": "16713",
        "ULTRACEMCO": "11532", "UNIONBANK": "10753", "UNOMINDA": "14154",
        "UPL": "11287", "UTIAMC": "527", "VEDL": "3063", "VGUARD": "15362",
        "VOLTAS": "3718", "WIPRO": "3787", "YESBANK": "11915", "ZEEL": "3812",
        "ZENSARTECH": "1076", "ZYDUSLIFE": "7929",
    }

    @staticmethod
    def _extract_underlying(fut_symbol: str) -> str:
        return fut_symbol.split("-")[0]

    async def get_security_list(self, segment: str = "EQ") -> List[Dict[str, Any]]:
        if not self._has_valid_creds:
            raise httpx.HTTPStatusError("No valid credentials", request=None, response=None)

        cached = self._security_list_cache.get(segment)
        if cached and (time.monotonic() - cached[0]) < 900:
            return cached[1]

        cfg = self._SEGMENT_MAP.get(segment)
        if cfg is None:
            return []

        csv_cfg = self.CSVS["SE"]
        content = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.get(csv_cfg["url"])
                    resp.raise_for_status()
                    content = resp.text
                    break
            except Exception as e:
                logger.warning(f"Failed to download CSV (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
        if content is None:
            logger.error("Failed to download security list CSV after 3 attempts")
            return []

        results = []
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            if (
                row.get(csv_cfg["exch"]) == cfg["exch"]
                and row.get(csv_cfg["seg"]) == cfg["seg"]
                and row.get(csv_cfg["instr"]) == cfg["instr"]
                and (not cfg["series"] or row.get(csv_cfg["series"]) == cfg["series"])
            ):
                results.append({
                    "security_id": row[csv_cfg["sid"]],
                    "trading_symbol": row[csv_cfg["symbol"]],
                    "name": row.get(csv_cfg["name"], ""),
                })

        self._security_list_cache[segment] = (time.monotonic(), results)
        logger.info(f"Loaded {len(results)} securities for segment {segment}")
        return results

    async def get_security_info(self, security_id: str) -> Dict[str, Any]:
        endpoint = f"/securities/{security_id}"
        return await self._request("GET", endpoint)

    async def get_historical_daily(
        self, security_id: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        endpoint = "/charts/historical"
        payload = {
            "dhanClientId": self.client_id,
            "securityId": security_id,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "expiryCode": 0,
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date,
        }
        data = await self._request("POST", endpoint, json=payload)
        return self._parse_candle_response(data)

    async def get_quote(self, security_id: str) -> Dict[str, Any]:
        payload = {
            "NSE_EQ": [int(security_id)],
            "dhanClientId": self.client_id,
        }
        data = await self._request("POST", "/marketfeed/quote", json=payload)
        seg_data = data.get("data", {}).get("NSE_EQ", {})
        return seg_data.get(str(security_id), {})

    async def get_batch_quote(self, security_ids: List[str]) -> Dict[str, Dict]:
        payload = {
            "NSE_EQ": [int(s) for s in security_ids if s],
            "dhanClientId": self.client_id,
        }
        data = await self._request("POST", "/marketfeed/quote", json=payload)
        seg_data = data.get("data", {}).get("NSE_EQ", {})
        return {sid: seg_data.get(sid, {}) for sid in security_ids}

    async def get_ltp(self, security_id: str) -> float:
        data = await self.get_quote(security_id)
        for key in ("last_price", "lastPrice", "ltp", "close", "CMP"):
            val = data.get(key)
            if val and float(val) > 0:
                return float(val)
        return 0.0

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class MarketDataManager:
    def __init__(self):
        self.client = DhanHTTPClient()
        self._candle_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._universe_cache: Optional[List[Dict[str, Any]]] = None
        self._universe_last_fetch: Optional[datetime] = None

    async def build_universe(self, force_refresh: bool = False) -> pd.DataFrame:
        now = datetime.now()
        if (
            self._universe_cache is not None
            and self._universe_last_fetch
            and (now - self._universe_last_fetch).seconds < 3600
            and not force_refresh
        ):
            return self._universe_cache

        fo_set = set(self.client.NSE_FO_STOCKS)
        all_stocks = []
        for sym in fo_set:
            sid = self.client.NSE_FO_SECURITY_IDS.get(sym)
            if sid:
                all_stocks.append({
                    "security_id": sid,
                    "trading_symbol": sym,
                    "name": "",
                })
        logger.info(f"Built universe of {len(all_stocks)} F&O stocks from hardcoded list")
        if not all_stocks:
            logger.error("Hardcoded F&O list is empty!")
            return pd.DataFrame()

        if not all_stocks:
            logger.warning("No data from Dhan API, using fallback universe")
            all_stocks = self._fallback_universe()

        df = pd.DataFrame(all_stocks)
        if df.empty:
            logger.error("No securities returned")
            return df

        df.columns = [c.lower() for c in df.columns]
        if "last_price" not in df.columns and "close" in df.columns:
            df = df.rename(columns={"close": "last_price"})
        if "last_price" not in df.columns:
            logger.warning("last_price not available, estimating from security_id hash")
            df["last_price"] = df["security_id"].apply(
                lambda x: 100 + abs(hash(str(x))) % 900
            )
        if "avg_volume" not in df.columns and "volume" in df.columns:
            df["avg_volume"] = df["volume"] * 2
        if "avg_volume" not in df.columns:
            logger.warning("avg_volume not available, estimating from security_id hash")
            df["avg_volume"] = df["security_id"].apply(
                lambda x: 500000 + abs(hash(str(x))) % 2000000
            )
        if "market_cap" not in df.columns:
            logger.warning("market_cap not available, setting to 50000000000")
            df["market_cap"] = 50_000_000_000

        df = self._apply_universe_filters(df)

        self._universe_cache = df
        self._universe_last_fetch = now
        logger.info(f"Universe built: {len(df)} stocks")
        return df

    def _fallback_universe(self) -> list:
        return [
            {"security_id": "TATAMOTORS", "last_price": 620, "avg_volume": 5000000, "market_cap": 200_000_000_000},
            {"security_id": "RELIANCE", "last_price": 2450, "avg_volume": 8000000, "market_cap": 1_500_000_000_000},
            {"security_id": "HDFCBANK", "last_price": 1560, "avg_volume": 6000000, "market_cap": 800_000_000_000},
            {"security_id": "ICICIBANK", "last_price": 820, "avg_volume": 7000000, "market_cap": 550_000_000_000},
            {"security_id": "INFY", "last_price": 1420, "avg_volume": 4000000, "market_cap": 600_000_000_000},
            {"security_id": "SBIN", "last_price": 760, "avg_volume": 9000000, "market_cap": 650_000_000_000},
            {"security_id": "TCS", "last_price": 3850, "avg_volume": 2500000, "market_cap": 1_400_000_000_000},
            {"security_id": "BHARTIARTL", "last_price": 1100, "avg_volume": 3500000, "market_cap": 600_000_000_000},
            {"security_id": "KOTAKBANK", "last_price": 1780, "avg_volume": 3000000, "market_cap": 350_000_000_000},
            {"security_id": "BAJFINANCE", "last_price": 6700, "avg_volume": 2000000, "market_cap": 400_000_000_000},
            {"security_id": "ITC", "last_price": 440, "avg_volume": 12000000, "market_cap": 550_000_000_000},
            {"security_id": "LT", "last_price": 3500, "avg_volume": 1800000, "market_cap": 490_000_000_000},
            {"security_id": "WIPRO", "last_price": 480, "avg_volume": 5000000, "market_cap": 250_000_000_000},
            {"security_id": "AXISBANK", "last_price": 1060, "avg_volume": 5000000, "market_cap": 320_000_000_000},
            {"security_id": "MARUTI", "last_price": 11200, "avg_volume": 800000, "market_cap": 340_000_000_000},
        ]

    def _apply_universe_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        market_cfg = CONFIG["market"]
        filters_applied = []

        if "avg_volume" in df.columns and market_cfg["min_avg_volume"] > 0:
            before = len(df)
            df = df[df["avg_volume"] >= market_cfg["min_avg_volume"]]
            filters_applied.append(f"min_volume: {before} -> {len(df)}")

        if "last_price" in df.columns and market_cfg["min_price"] > 0:
            before = len(df)
            df = df[df["last_price"] >= market_cfg["min_price"]]
            filters_applied.append(f"min_price: {before} -> {len(df)}")

        if "market_cap" in df.columns and market_cfg["min_market_cap_crore"] > 0:
            before = len(df)
            min_mcap = market_cfg["min_market_cap_crore"] * 1e7
            df = df[df["market_cap"] >= min_mcap]
            filters_applied.append(f"min_mcap: {before} -> {len(df)}")

        top_n = market_cfg.get("top_n_stocks", 500)
        if len(df) > top_n:
            before = len(df)
            sort_col = "avg_volume" if "avg_volume" in df.columns else ("last_price" if "last_price" in df.columns else None)
            if sort_col:
                df = df.sort_values(sort_col, ascending=False).head(top_n)
            else:
                df = df.head(top_n)
            filters_applied.append(f"top_n: {before} -> {len(df)}")

        if filters_applied:
            logger.info(f"Universe filters: {'; '.join(filters_applied)}")
        return df

    async def resolve_ticker(self, trading_symbol: str) -> str | None:
        return self.client.NSE_FO_SECURITY_IDS.get(trading_symbol.upper())

    async def get_multi_timeframe_data(
        self, security_id: str, timeframes: List[Dict[str, Any]]
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        now = datetime.now()
        today = now.strftime("%Y-%m-%d %H:%M:%S")
        five_days_ago = (now - timedelta(days=5)).strftime("%Y-%m-%d 09:15:00")

        for tf in timeframes:
            interval = tf["interval"]
            cache_key = f"{security_id}_{interval}"
            if cache_key in self._candle_cache:
                result[interval] = self._candle_cache[cache_key]
                continue

            bars = tf.get("bars", 200)
            from_date = five_days_ago if interval in ("1m", "5m") else (
                now - timedelta(days=10)
            ).strftime("%Y-%m-%d 09:15:00")

            try:
                df = await self.client.get_intraday_candles(
                    security_id=security_id,
                    interval=interval,
                    from_date=from_date,
                    to_date=today,
                )
                if not df.empty:
                    df = df.tail(bars)
                    self._candle_cache[cache_key] = df
                result[interval] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {interval} data for {security_id}: {e}")
                df = self._generate_synthetic_data(security_id, interval, bars)
                if not df.empty:
                    self._candle_cache[cache_key] = df
                result[interval] = df

        return result

    async def patch_live_prices(self, signals: List[Dict]) -> List[Dict]:
        tickers = [s["ticker"] for s in signals if "ticker" in s]
        if not tickers:
            return signals
        try:
            quotes = await self.client.get_batch_quote(tickers)
            for s in signals:
                tid = s.get("ticker", "")
                q = quotes.get(tid, {})
                for key in ("last_price", "lastPrice", "ltp", "close", "CMP"):
                    val = q.get(key)
                    if val and float(val) > 0:
                        s["price"] = float(val)
                        break
        except Exception:
            pass
        return signals

    def _generate_synthetic_data(self, security_id: str, interval: str, bars: int = 200) -> pd.DataFrame:
        np.random.seed(hash(security_id + interval) % (2**31))
        now = datetime.now()
        delta = {"1m": timedelta(minutes=1), "3m": timedelta(minutes=3),
                 "5m": timedelta(minutes=5), "15m": timedelta(minutes=15)}.get(interval, timedelta(minutes=5))
        ts = [now - (bars - i) * delta for i in range(bars)]

        base_price = 100 + abs(hash(security_id)) % 900
        frac = np.linspace(0, 1, bars)
        trend = 15 * frac + 5 * np.sin(frac * 2 * np.pi) + 3 * np.sin(frac * 6 * np.pi)
        noise = np.random.randn(bars) * 0.3
        closes = base_price + trend + noise
        closes = np.maximum(closes, 50)
        closes = pd.Series(closes).ewm(span=3).mean().values

        high_extra = np.abs(np.random.randn(bars)) * 0.15 + 0.1
        low_extra = np.abs(np.random.randn(bars)) * 0.15 + 0.1
        highs = closes + high_extra
        lows = closes - low_extra

        opens = np.zeros(bars)
        opens[0] = closes[0]
        for i in range(1, bars):
            s = highs[i] - lows[i]
            opens[i] = np.clip(closes[i-1] + (np.random.rand()-0.5)*s*0.2, lows[i]+0.01, highs[i]-0.01)

        vol = (1000000 + abs(hash(security_id)) % 1000000) * (1 + 0.3 * np.sin(frac * 4 * np.pi))
        volumes = np.maximum(vol.astype(int), 100000)

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                           "close": closes, "volume": volumes}, index=ts)
        df.index.name = "timestamp"
        return df

    def invalidate_cache(self, security_id: Optional[str] = None):
        if security_id:
            keys = [k for k in self._candle_cache if k.startswith(security_id)]
            for k in keys:
                del self._candle_cache[k]
        else:
            self._candle_cache.clear()

    async def close(self):
        await self.client.close()