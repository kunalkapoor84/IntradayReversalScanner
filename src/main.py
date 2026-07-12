from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.alerts.alerts import AlertManager
from src.backtest.engine import BacktestEngine
from src.catalysts.catalysts import CatalystDetector, MarketContext
from src.config import CONFIG
from src.dashboard.dashboard import LiveDashboard
from src.data.dhan_client import MarketDataManager
from src.data.universe import UniverseBuilder
from src.ranking.scoring import RankerEngine, ConfidenceScorer
from src.risk.manager import RiskManager
from src.scanners.manager import ScannerManager


class ScannerOrchestrator:
    def __init__(self, use_dashboard=True):
        self.market_data = MarketDataManager()
        self.scanner_manager = ScannerManager(self.market_data)
        self.alert_manager = AlertManager()
        self.use_dashboard = use_dashboard
        self.dashboard = LiveDashboard(refresh_seconds=CONFIG["dashboard"]["refresh_seconds"]) if use_dashboard else None
        self.backtest = BacktestEngine()
        self.catalyst_detector = CatalystDetector()
        self.market_context = MarketContext()
        self._running = False
        self._cycle_count = 0
        self._initialized = False
        self._name_map = {}
        self._all_signals_log: List[Dict] = []

    def _describe_signal(self, s: dict) -> str:
        parts = []
        signal = s.get("signal", "")
        direction = s.get("direction", "")
        tfs = s.get("timeframe_signals", {})
        ind = s.get("indicators", {})
        vol = s.get("volume", {})
        smc = tfs.get("smc", [])
        pat_5m = tfs.get("5m_patterns", tfs.get("patterns_5m", []))
        pat_1m = tfs.get("patterns_1m", [])
        pats_5m = [p for p in pat_5m if p] if isinstance(pat_5m, list) else []
        pats_1m = [p for p in pat_1m if p] if isinstance(pat_1m, list) else []

        # Candlestick patterns (1m entry timeframe)
        if pats_1m:
            parts.append(f"1m candle: {', '.join(pats_1m[:3])}")

        # Chart patterns / SMC (5m)
        if smc:
            parts.append(f"SMC(5m): {', '.join(smc[:2])}")

        if "Bullish Pullback" in signal:
            parts.append("Pullback to support, 15m EMA bullish aligned")
        elif "Bearish Pullback" in signal:
            parts.append("Pullback rally into resistance, 15m EMA bearish aligned")
        elif "Exhaustion Reversal" in signal:
            if direction == "long":
                parts.append("Selling climax, long lower wick / reversal candle")
            else:
                parts.append("Buying climax, long upper wick / rejection candle")
            if tfs.get("vol_climax"): parts.append("Volume climax")
            if tfs.get("vwap_deviation_atr", 0) >= 2: parts.append(f"Stretched {tfs['vwap_deviation_atr']:.1f}ATR from VWAP")
        elif "Trend Reversal" in signal:
            triggers = []
            if tfs.get("bullish_cross") or tfs.get("bearish_cross"): triggers.append("EMA 9/20 crossover")
            if tfs.get("vwap_reclaim"): triggers.append("VWAP reclaim")
            if tfs.get("rsi_cross"): triggers.append("RSI crossed 50")
            parts.append(" | ".join(triggers[:3]) if triggers else "Multiple reversal signals")
        elif "VWAP Reversal" in signal:
            deviation = tfs.get("vwap_distance_atr", 0)
            parts.append(f"Price {deviation:.1f}ATR from VWAP")
            if tfs.get("volume_climax"): parts.append("Volume climax")
        elif "Failed Breakdown" in signal:
            parts.append("Broke below support, recovered — fakeout")
            if vol.get("is_spike"): parts.append("Volume spike")
        elif "Failed Breakout" in signal:
            parts.append("Broke above resistance, rejected — fakeout")
            if vol.get("is_spike"): parts.append("Volume spike")

        # Indicators (all from 5m timeframe unless labeled)
        ind_lines = []
        rsi_5 = ind.get("rsi_5", 0)
        rsi_15 = ind.get("rsi_15", 0)
        adx_5 = ind.get("adx_5", 0)
        atr_5 = ind.get("atr_5", 0)
        macd_5 = ind.get("macd_5", 0)
        macd_h = ind.get("macd_hist_5", 0)
        if rsi_5: ind_lines.append(f"RSI(5m)={rsi_5:.0f}" + (" (bull)" if rsi_5 > 50 else " (bear)"))
        if rsi_15: ind_lines.append(f"RSI(15m)={rsi_15:.0f}")
        if adx_5: ind_lines.append(f"ADX(5m)={adx_5:.0f}" + (" trending" if adx_5 >= 25 else ""))
        if atr_5: ind_lines.append(f"ATR(5m)={atr_5:.1f}")
        if macd_5: ind_lines.append(f"MACD(5m)={macd_5:.2f}")
        if macd_h: ind_lines.append(f"Hist(5m)={'+' if macd_h > 0 else '-'}{abs(macd_h):.2f}")
        vol_ratio = vol.get("volume_ratio", 0)
        if vol_ratio: ind_lines.append(f"Vol(5m)={vol_ratio:.1f}x")
        parts.extend(ind_lines[:4])

        return " | ".join(parts) if parts else signal

    def _print_signals(self, signals):
        sep = "=" * 80
        print(f"\n{sep}")
        print(f"SCAN CYCLE #{self._cycle_count} @ {datetime.now().strftime('%H:%M:%S')}")
        print(sep)
        if not signals:
            print("No signals found")
            return
        header = f"{'Rank':<5} {'Ticker':<8} {'Signal':<28} {'Dir':<5} {'Conf':<6} {'RR':<5} {'Entry':<10} {'Stop':<10} {'Quality':<8}"
        print(header)
        print("-" * 80)
        for i, s in enumerate(signals):
            ticker = s.get('ticker', '')
            signal = s.get('signal', '')[:25]
            direction = s.get('direction', '')
            conf = s.get('confidence', 0)
            risk = s.get('risk_management', {})
            rr = risk.get('risk_reward', 0)
            entry = risk.get('entry', '-')
            stop = risk.get('stop_loss', '-')
            quality = s.get('quality', '-')
            desc = self._describe_signal(s)
            print(f"{i+1:<5} {ticker:<8} {signal:<20} {direction:<5} {conf:<6} {rr:<5} {str(entry):<10} {str(stop):<10} {quality:<8}")
            print(f"      {desc}")
        print(f"{sep}\n")

    async def initialize(self):
        if self._initialized:
            return
        logger.info("Initializing scanner orchestrator...")
        self.universe_builder = UniverseBuilder(self.market_data)
        df = await self.market_data.build_universe()
        universe_size = len(df) if isinstance(df, pd.DataFrame) else 0
        logger.info(f"Universe size: {universe_size}")
        await self.market_context.update()
        self._initialized = True
        logger.info("Initialization complete")

    async def _get_universe(self) -> pd.DataFrame:
        df = await self.market_data.build_universe()
        if not df.empty and "trading_symbol" in df.columns:
            self._name_map = dict(zip(df["security_id"], df["trading_symbol"]))
        else:
            self._name_map = {}
        return df

    async def _get_tickers(self) -> List[str]:
        df = await self._get_universe()
        if not df.empty:
            if "security_id" in df.columns:
                return df["security_id"].tolist()
            return df.index.tolist()
        return []

    async def run_live(self):
        await self.initialize()
        self._running = True
        signal_queue: asyncio.Queue = asyncio.Queue()

        dashboard_task = None
        if self.use_dashboard:
            dashboard_task = asyncio.create_task(self.dashboard.run(signal_queue))

        logger.info("Starting live scanning loop...")
        while self._running:
            try:
                cycle_start = time.time()
                self._cycle_count += 1
                logger.info(f"=== Scan Cycle #{self._cycle_count} ===")

                tickers = await self._get_tickers()

                if not tickers:
                    logger.warning("No tickers to scan")
                    await asyncio.sleep(30)
                    continue

                signals = await self.scanner_manager.scan_all(tickers)
                signals = await self.market_data.patch_live_prices(signals)

                for s in signals:
                    try:
                        risk = RiskManager(s)
                        risk_data = risk.get_all()
                        s["risk_management"] = risk_data
                        s["quality"] = risk_data.get("quality", "B")
                        s["quality_label"] = risk_data.get("quality", "B")
                        s["name"] = self._name_map.get(s.get("ticker", ""), s.get("ticker", ""))
                    except Exception as e:
                        logger.debug(f"Risk calc error for {s.get('ticker')}: {e}")

                try:
                    ranked_signals = []
                    for s in signals:
                        clean = {}
                        for k, v in s.items():
                            if isinstance(v, dict):
                                clean[k] = {}
                                for sk, sv in v.items():
                                    if isinstance(sv, pd.Series):
                                        sv = float(sv.iloc[-1]) if not sv.empty else 0
                                    elif hasattr(sv, 'item') and not isinstance(sv, pd.Series):
                                        sv = float(sv)
                                    clean[k][sk] = sv
                            elif isinstance(v, pd.Series):
                                clean[k] = float(v.iloc[-1]) if not v.empty else 0
                            elif hasattr(v, 'item') and not isinstance(v, pd.Series):
                                clean[k] = float(v)
                            elif isinstance(v, list):
                                clean[k] = [str(x) for x in v]
                            else:
                                clean[k] = v
                        ranked_signals.append(clean)
                    ranked_signals = RankerEngine(ranked_signals).rank(top_n=10)
                except Exception as e:
                    logger.error(f"Ranker error: {e}")
                    ranked_signals = signals[:10]

                await self.market_context.update()

                if self.use_dashboard:
                    self.dashboard.update_market_context(self.market_context.get_summary())
                    await signal_queue.put(ranked_signals)
                else:
                    self._print_signals(ranked_signals)

                for s in ranked_signals:
                    await self.alert_manager.send_alert(s)

                elapsed = time.time() - cycle_start
                refresh = CONFIG["dashboard"]["refresh_seconds"]
                sleep_time = max(1, refresh - elapsed)

                for s in signals:
                    desc = self._describe_signal(s)
                    logger.info(
                        f"  SIGNAL: {s.get('name','?'):>12} ({s.get('ticker')}) "
                        f"| {s.get('direction','?').upper():>5} "
                        f"| {s.get('signal','?'):>20} | conf={s.get('confidence',0):.0f} "
                        f"| rr={s.get('risk_management',{}).get('risk_reward',0):.1f}"
                    )
                    logger.info(f"          {desc}")

                try:
                    self._all_signals_log.extend(signals)
                    self._save_signals_excel()
                except Exception as e:
                    logger.debug(f"Excel save error: {e}")

                logger.info(
                    f"Cycle complete: {len(signals)} raw, {len(ranked_signals)} ranked [{elapsed:.1f}s]"
                )
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scan cycle error: {e}")
                await asyncio.sleep(10)

        if dashboard_task:
            dashboard_task.cancel()
        await self.shutdown()

    def _save_signals_excel(self):
        if not self._all_signals_log:
            return
        records = []
        for s in self._all_signals_log:
            risk = s.get("risk_management", {})
            ind = s.get("indicators", {})
            vol = s.get("volume", {})
            tfs = s.get("timeframe_signals", {})
            records.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "security_id": s.get("ticker", ""),
                "name": s.get("name", ""),
                "signal": s.get("signal", ""),
                "direction": s.get("direction", ""),
                "confidence": s.get("confidence", 0),
                "price": s.get("price", 0),
                "entry": risk.get("entry", 0),
                "stop_loss": risk.get("stop_loss", 0),
                "target_1": risk.get("targets", {}).get("target_1", 0),
                "target_2": risk.get("targets", {}).get("target_2", 0),
                "target_3": risk.get("targets", {}).get("target_3", 0),
                "risk_reward": risk.get("risk_reward", 0),
                "quality": s.get("quality", ""),
                "scanner": s.get("scanner", ""),
                "volume_ratio": vol.get("volume_ratio", 0),
                "rsi_5": ind.get("rsi_5", 0),
                "adx_5": ind.get("adx_5", 0),
                "atr_5": ind.get("atr_5", 0),
                "macd_5": ind.get("macd_5", 0),
                "macd_signal_5": ind.get("macd_signal_5", 0),
                "macd_hist_5": ind.get("macd_hist_5", 0),
                "vwap_distance": s.get("vwap_distance_5", s.get("vwap_distance_5m", 0)),
                "sector": s.get("sector", ""),
                "description": self._describe_signal(s),
            })
        df = pd.DataFrame(records)
        path = Path("signals_output.xlsx")
        df.to_excel(path, index=False, engine="openpyxl")
        logger.info(f"Signals saved to {path} ({len(df)} rows)")

    async def run_backtest(self, years: int = None):
        await self.initialize()
        years = years or CONFIG["backtest"]["years"]
        logger.info(f"Running backtest for {years} years...")

        tickers = await self._get_tickers()
        tickers = tickers[:200]
        logger.info(f"Backtest universe: {len(tickers)} stocks")

        signals = await self.scanner_manager.scan_all(tickers)

        price_data = pd.DataFrame()
        if tickers:
            cache_key = f"{tickers[0]}_5m"
            price_data = self.market_data._candle_cache.get(cache_key, pd.DataFrame())

        results = self.backtest.run(signals, price_data)
        logger.success(f"Backtest complete: {results}")
        return results

    async def shutdown(self):
        self._running = False
        await self.market_data.close()
        await self.alert_manager.close()
        logger.info("Scanner shut down gracefully")

    async def run_single_ticker(self, ticker: str):
        await self.initialize()
        logger.info(f"Scanning single ticker: {ticker}")
        tf_data = await self.market_data.get_multi_timeframe_data(
            ticker, CONFIG["timeframes"]
        )
        enabled = [s for s, c in CONFIG["scanners"].items()
                   if isinstance(c, dict) and c.get("enabled", False)]
        signals = self.scanner_manager._scan_single(ticker, tf_data, enabled)

        for s in signals:
            risk = RiskManager(s)
            s["risk_management"] = risk.get_all()
            rm = s["risk_management"]
            tfs = s.get("timeframe_signals", {})
            ind = s.get("indicators", {})
            vol = s.get("volume", {})
            print(f"\n{'='*60}")
            print(f"Ticker: {s['ticker']}")
            print(f"Signal: {s['signal']} ({s['direction'].upper()})")
            print(f"Confidence: {s['confidence']}/100")
            print(f"Entry: {rm['entry']} | Stop: {rm['stop_loss']}")
            print(f"Targets: T1={rm['targets']['target_1']} T2={rm['targets']['target_2']} T3={rm['targets']['target_3']}")
            print(f"Risk/Reward: {rm['risk_reward']} | Probability: {rm['probability']}%")
            print(f"Quality: {rm['quality']}")
            print(f"Volume Ratio: {vol.get('volume_ratio', 'N/A')}x")
            print(f"RSI: {ind.get('rsi_5', 'N/A')}")
            print(f"ADX: {ind.get('adx_5', 'N/A')}")
            print(f"Patterns (1m): {tfs.get('patterns_1m', [])}")
            print(f"SMC: {tfs.get('smc', [])}")
            print(f"Catalysts: {s.get('catalysts', [])}")
            print(f"Quality Label: {s.get('quality_label', 'N/A')}")
            print("="*60)

        return signals


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Intraday Reversal Scanner")
    parser.add_argument("--mode", choices=["live", "backtest", "single"], default="live")
    parser.add_argument("--ticker", type=str, help="Single ticker for --mode single")
    parser.add_argument("--years", type=int, default=5, help="Backtest years")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable Rich dashboard, print to console")
    args = parser.parse_args()

    orch = ScannerOrchestrator(use_dashboard=not args.no_dashboard)
    try:
        if args.mode == "live":
            await orch.run_live()
        elif args.mode == "backtest":
            await orch.run_backtest(args.years)
        elif args.mode == "single" and args.ticker:
            await orch.run_single_ticker(args.ticker)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        await orch.shutdown()


if __name__ == "__main__":
    asyncio.run(main())