from __future__ import annotations

from typing import Any, Dict, List

from src.config import CONFIG


class ConfidenceScorer:
    def __init__(self, signal: Dict[str, Any]):
        self.signal = signal
        self.weights = CONFIG["confidence"]

    def score(self) -> int:
        return self.signal.get("confidence", 50)

    def components(self) -> Dict[str, float]:
        s = self.signal
        return {
            "trend_quality": min(20, s.get("confidence", 0) * 0.2),
            "volume": min(20, self._volume_score()),
            "structure": min(20, self._structure_score()),
            "momentum": min(15, s.get("indicators", {}).get("adx_5", 0) * 0.75),
            "institutional": min(15, self._institutional_score()),
            "relative_strength": min(10, self._rs_score()),
        }

    def _volume_score(self) -> float:
        vol = self.signal.get("volume", {})
        score = 0
        vr = vol.get("volume_ratio", 1)
        if vol.get("is_spike", False): score += 10
        if vol.get("is_absorption", False): score += 8
        if vol.get("above_20dma", False): score += 5
        if vol.get("is_stopping", False): score += 7
        if vol.get("is_climax", False): score += 5
        return min(20, score)

    def _structure_score(self) -> float:
        tfs = self.signal.get("timeframe_signals", {})
        score = 0
        smc = tfs.get("smc", [])
        pat = tfs.get("patterns_1m", [])
        if "change_of_character" in smc: score += 8
        if "break_of_structure" in smc: score += 6
        if "liquidity_sweep" in smc: score += 5
        if "fair_value_gap" in smc: score += 4
        if "bullish_engulfing" in pat or "bearish_engulfing" in pat: score += 5
        if "hammer" in pat or "shooting_star" in pat: score += 4
        if "morning_star" in pat or "evening_star" in pat: score += 6
        if tfs.get("rsi_divergence", False): score += 5
        return min(20, score)

    def _institutional_score(self) -> float:
        vol = self.signal.get("volume", {})
        score = 0
        if vol.get("delivery_pct_increasing", False): score += 5
        if vol.get("above_20dma", False): score += 3
        if vol.get("is_absorption", False): score += 7
        return min(15, score)

    def _rs_score(self) -> float:
        rs = self.signal.get("relative_strength", 50)
        if rs >= 60: return 10
        if rs >= 50: return 7
        if rs >= 40: return 5
        return 2

    def grade(self) -> str:
        score = self.signal.get("confidence", 0)
        if score >= 92: return "A+"
        if score >= 85: return "A"
        if score >= 75: return "B"
        return "C"


class RankerEngine:
    def __init__(self, signals: List[Dict[str, Any]]):
        self.signals = signals

    def rank(self, top_n: int = 10) -> List[Dict[str, Any]]:
        scored = []
        for s in self.signals:
            scorer = ConfidenceScorer(s)
            components = scorer.components()
            quality = scorer.grade()
            ranked_signal = {
                **s,
                "score_components": components,
                "quality": quality,
                "rank_score": sum(components.values()),
            }
            scored.append(ranked_signal)

        scored.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        top = scored[:top_n]

        for i, s in enumerate(top):
            s["rank"] = i + 1

        return top