from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import CONFIG
from src.risk.manager import RiskManager


class BacktestEngine:
    def __init__(self):
        self.bt_config = CONFIG["backtest"]
        self.initial_capital = self.bt_config["initial_capital"]
        self.commission = self.bt_config["commission_pct"] / 100
        self.slippage = self.bt_config["slippage_pct"] / 100
        self.results: List[Dict] = []
        self.output_dir = Path(self.bt_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, signals: List[Dict[str, Any]], price_data: pd.DataFrame) -> Dict[str, Any]:
        capital = self.initial_capital
        trades = []
        peak_capital = capital
        drawdowns = []

        for signal in signals:
            risk = RiskManager(signal, capital)
            risk_data = risk.get_all()
            rr = risk_data["risk_reward"]

            if rr < CONFIG["risk"]["min_risk_reward"]:
                continue

            pos = risk_data["position"]
            entry = risk_data["entry"]
            sl = risk_data["stop_loss"]
            targets = risk_data["targets"]

            cost = pos["shares"] * entry
            if cost * (1 + self.commission) > capital:
                pos["shares"] = int(capital / (entry * (1 + self.commission)))
                cost = pos["shares"] * entry

            direction = signal.get("direction", "long")
            direction_mult = 1 if direction == "long" else -1

            trade = {
                "entry_time": datetime.now(),
                "exit_time": None,
                "direction": direction,
                "entry": entry,
                "stop_loss": sl,
                "targets": targets,
                "shares": pos["shares"],
                "capital_risk": cost * self.commission,
                "exit_price": None,
                "pnl": 0,
                "pnl_pct": 0,
                "rr_achieved": 0,
                "held_bars": 0,
                "win": None,
            }

            exit_price, exit_reason = self._simulate_exit(
                entry, sl, targets, direction_mult, price_data
            )

            if exit_reason == "stop_loss":
                trade_result["win"] = False
            elif exit_reason == "target":
                trade_result["win"] = True
            elif exit_reason == "time_exit":
                trade_result["win"] = direction_mult * (exit_price - entry) > 0
            else:
                trade_result["win"] = None

            trade_result["exit_price"] = exit_price
            pnl = direction_mult * (exit_price - entry) * pos["shares"]
            pnl -= cost * self.commission
            trade_result["pnl"] = round(pnl, 2)
            trade_result["pnl_pct"] = round(pnl / capital * 100, 2)
            trade_result["exit_reason"] = exit_reason

            capital += pnl
            peak_capital = max(peak_capital, capital)
            dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0
            drawdowns.append(dd)

            trades.append(trade_result)

        self.results = trades

        if not trades:
            return {"error": "No trades", "total_trades": 0}

        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wins = [t for t in trades if t.get("win") is True]
        losses = [t for t in trades if t.get("win") is False]
        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        profit_factor = abs(
            sum(t["pnl"] for t in wins) / sum(abs(t["pnl"]) for t in losses)
        ) if losses and sum(abs(t["pnl"]) for t in losses) > 0 else float("inf")
        avg_rr = np.mean([t.get("exit_risk_reward", 0) for t in trades]) if trades else 0
        max_dd = max(drawdowns) if drawdowns else 0
        expectancy = total_pnl / total_trades if total_trades > 0 else 0
        avg_hold = np.mean([t.get("held_bars", 0) for t in trades]) * 5 if trades else 0
        cagr = ((capital / self.initial_capital) ** (1 / max(1, self.bt_config["years"])) - 1) * 100

        returns = [t["pnl_pct"] for t in trades]
        sharpe = (
            (np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0)
            if returns
            else 0
        )

        result = {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "expectancy": round(expectancy, 2),
            "avg_rr": round(avg_rr, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "cagr_pct": round(cagr, 2),
            "avg_hold_minutes": round(avg_hold, 0),
            "total_pnl": round(total_pnl, 2),
            "final_capital": round(capital, 2),
            "initial_capital": self.initial_capital,
        }

        self._save_results(result, trades)
        handler.info(f"Backtest complete: {result}")
        return result

    def _simulate_exit(
        self, entry: float, sl: float, targets: Dict, direction: str, price_data: pd.DataFrame
    ) -> tuple:
        if price_data.empty or len(price_data) < 5:
            return entry, "no_data"

        direction_mult = 1 if direction == "long" else -1
        max_bars = min(78, len(price_data))  # ~6.5 hours

        for i in range(1, max_bars):
            candle = price_data.iloc[i]
            high = candle["high"]
            low = candle["low"]
            close = candle["close"]

            if direction == "long":
                if low <= sl:
                    return sl - (sl * self.slippage), "stop_loss"
                if high >= targets.get("target_1", entry * 1.02):
                    return min(high * (1 - self.slippage), targets["target_1"]), "target"
            else:
                if high >= sl:
                    return sl + (sl * self.slippage), "stop_loss"
                if low <= targets.get("target_1", entry * 0.98):
                    return max(low * (1 + self.slippage), targets["target_1"]), "target"

        return price_data.iloc[-1]["close"], "time_exit"

    def _save_results(self, summary: Dict, trades: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"backtest_summary_{timestamp}.json"
        trades_path = self.output_dir / f"backtest_trades_{timestamp}.csv"

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        pd.DataFrame(trades).to_csv(trades_path, index=False)
        logger.info(f"Backtest results saved to {self.output_dir}")