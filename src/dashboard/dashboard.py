from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from rich.console import Console
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import CONFIG


class LiveDashboard:
    def __init__(self, refresh_seconds: int = 30):
        self.console = Console()
        self.refresh = refresh_seconds
        self._running = False
        self._signals: List[Dict[str, Any]] = []
        self._market_context: Dict[str, Any] = {}
        self._last_update: str = ""

    def update_signals(self, signals: List[Dict[str, Any]]):
        self._signals = signals[:CONFIG["dashboard"]["max_signals_display"]]
        self._last_update = datetime.now().strftime("%H:%M:%S")

    def update_market_context(self, context: Dict[str, Any]):
        self._market_context = context

    def _build_table(self) -> Table:
        table = Table(
            title=f"🔍 Intraday Reversal Scanner",
            title_style="bold cyan",
            header_style="bold white",
            border_style="blue",
            show_lines=True,
        )

        columns = [
            "Rank", "Ticker", "Signal", "Score", "Dir", "Sector",
            "Vol Rat", "RSI", "ADX", "Entry", "Stop", "RR", "Quality", "Time"
        ]
        for col in columns:
            table.add_column(col, justify="center" if col != "Signal" else "left")

        for i, s in enumerate(self._signals[:10]):
            rank = str(i + 1)
            ticker = s.get("ticker", "")
            signal = s.get("signal", "")[:20]
            score = s.get("confidence", 0)
            direction = s.get("direction", "")
            sector = s.get("sector", "")[:10]
            vol = s.get("volume", {})
            ind = s.get("indicators", {})
            risk = s.get("risk_management", {})
            tfs = s.get("timeframe_signals", {})

            vol_ratio = f"{vol.get('volume_ratio', '-')}x"
            rsi = str(ind.get("rsi_5", "-"))
            adx = str(ind.get("adx_5", "-"))
            entry = str(risk.get("entry", "-"))
            rr = str(risk.get("risk_reward", "-"))
            quality = s.get("quality", tfs.get("quality_label", "-"))
            now = self._last_update

            score_color = "green" if score >= 85 else "yellow" if score >= 75 else "red"
            dir_color = "green" if direction == "long" else "red"
            dir_symbol = "▲" if direction == "long" else "▼"

            table.add_row(
                rank,
                ticker,
                signal,
                Text(str(score), style=score_color),
                Text(dir_symbol, style=dir_color),
                sector, vol_ratio, rsi, adx,
                entry, rr,
                Text(quality, style="bold cyan" if quality == "A+" else "bold white"),
                now,
            )

        return table

    def _build_context_panel(self) -> Panel:
        ctx = self._market_context
        lines = []
        for k, v in ctx.items():
            k_str = k.replace("_", " ").title()
            lines.append(f"{k_str}: {v}")
        text = "\n".join(lines) if lines else "Loading market context..."
        return Panel(Text(text), title="📊 Market Context", border_style="green")

    def _build_stats_panel(self) -> Panel:
        total = len(self._signals)
        longs = sum(1 for s in self._signals if s.get("direction") == "long")
        shorts = total - longs
        avg_score = sum(s.get("confidence", 0) for s in self._signals) / max(total, 1)

        stats_text = (
            f"Total Signals: {total}\n"
            f"Long: {longs} | Short: {shorts}\n"
            f"Avg Score: {avg_score:.1f}\n"
            f"Updated: {self._last_update}\n"
            f"Refresh: {self.refresh}s"
        )
        return Panel(Text(stats_text), title="📈 Scanner Stats", border_style="blue")

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=3),
            Layout(name="main", ratio=2),
            Layout(name="bottom", size=6),
        )
        layout["top"].split_row(
            Layout(self._build_stats_panel()),
            Layout(self._build_market_context_panel()),
        )
        layout["main"].update(self._build_table())
        return layout

    async def run(self, signal_queue: asyncio.Queue):
        self._running = True
        top_sec = CONFIG["dashboard"].get("timeout", 300)

        with Live(self._build_layout(), refresh_per_second=2, screen=False) as live:
            while self._running:
                try:
                    signals = await asyncio.wait_for(
                        signal_queue.get(), timeout=self.refresh
                    )
                    if signals:
                        self.update_signals(signals)
                except asyncio.TimeoutError:
                    pass

                live.update(self._build_layout())
                await asyncio.sleep(0.5)

    def stop(self):
        self._running = False