from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

import pandas as pd

from src.config import CONFIG


class AlertManager:
    def __init__(self):
        self.config = CONFIG["alerts"]
        self._telegram_bot = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=10)
        return self._http_client

    async def send_alert(self, signal: Dict[str, Any]):
        if signal.get("confidence", 0) < CONFIG["confidence"]["alert_threshold"]:
            return

        message = self._format_message(signal)

        if self.config.get("desktop_notification", False):
            await self._send_desktop(signal, message)

        if self.config.get("sound", False):
            await self._play_sound()

        tasks = []
        if self.config.get("telegram", {}).get("enabled", False):
            tasks.append(self._send_telegram(message))
        if self.config.get("discord", {}).get("enabled", False):
            tasks.append(self._send_discord(message))
        if self.config.get("slack", {}).get("enabled", False):
            tasks.append(self._send_slack(message))
        if self.config.get("email", {}).get("enabled", False):
            tasks.append(self._send_email(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Alert sent: {signal.get('ticker')} {signal.get('signal')} @ {signal.get('confidence')}")

    def _format_message(self, signal: Dict[str, Any]) -> str:
        ticker = signal.get("ticker", "N/A")
        stype = signal.get("signal", "N/A")
        direction = signal.get("direction", "N/A")
        confidence = signal.get("confidence", 0)
        sector = signal.get("sector", "N/A")
        vol = signal.get("volume", {})
        indicators = signal.get("indicators", {})

        emoji = "🟢" if direction == "long" else "🔴"
        msg = (
            f"{emoji} *{ticker}* - {stype}\n"
            f"   Confidence: {confidence}/100\n"
            f"   Direction: {direction.upper()}\n"
            f"   Sector: {sector}\n"
            f"   Volume Ratio: {vol.get('volume_ratio', 'N/A')}x\n"
            f"   RSI: {indicators.get('rsi_5', 'N/A')}\n"
            f"   ADX: {indicators.get('adx_5', 'N/A')}\n"
            f"   Time: {pd.Timestamp.now().strftime('%H:%M:%S')}"
        )
        return msg

    async def _send_desktop(self, signal: Dict, message: str):
        try:
            from plyer import notification
            notification.notify(
                title=f"{signal.get('signal')} - {signal.get('ticker')}",
                message=message,
                timeout=10,
            )
        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")

    async def _play_sound(self):
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    async def _send_telegram(self, message: str):
        cfg = self.config["telegram"]
        if not cfg.get("bot_token") or not cfg.get("chat_id"):
            return
        try:
            url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
            client = await self._get_http()
            await client.post(
                url,
                json={"chat_id": cfg["chat_id"], "text": message, "parse_mode": "Markdown"},
            )
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")

    async def _send_discord(self, message: str):
        cfg = self.config["discord"]
        if not cfg.get("webhook_url"):
            return
        try:
            client = await self._get_http()
            await client.post(
                cfg["webhook_url"],
                json={"content": message},
            )
        except Exception as e:
            logger.warning(f"Discord alert failed: {e}")

    async def _send_slack(self, message: str):
        cfg = self.config["slack"]
        if not cfg.get("webhook_url"):
            return
        try:
            client = await self._get_http()
            await client.post(
                cfg["webhook_url"],
                json={"text": message},
            )
        except Exception as e:
            logger.warning(f"Slack alert failed: {e}")

    async def _send_email(self, message: str):
        cfg = self.config["email"]
        if not all([cfg.get("smtp_host"), cfg.get("from_addr"), cfg.get("to_addr")]):
            return
        try:
            import smtplib
            from email.message import EmailMessage
            msg = EmailMessage()
            msg.set_content(message)
            msg["Subject"] = "Intraday Reversal Scanner Alert"
            msg["From"] = cfg["from_addr"]
            msg["To"] = cfg["to_addr"]
            with smtplib.SMTP(cfg["smtp_host"], cfg.get("smtp_port", 587)) as server:
                server.starttls()
                if cfg.get("smtp_user"):
                    server.login(cfg["smtp_user"], cfg.get("smtp_password", ""))
                server.send_message(msg)
        except Exception as e:
            logger.warning(f"Email alert failed: {e}")

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()