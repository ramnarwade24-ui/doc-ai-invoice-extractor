from __future__ import annotations

from typing import Any, Dict, Tuple

from utils.config import PipelineConfig


def _clamp01(x: float) -> float:
	return max(0.0, min(1.0, float(x)))


def normalize_latency(latency_sec: float, target_sec: float) -> float:
	"""Convert latency to a 0..1 score where 1 is best (<= target)."""
	if target_sec <= 0:
		return 0.0
	return _clamp01(1.0 - (float(latency_sec) / float(target_sec)))


def normalize_cost(cost_usd: float, target_usd: float) -> float:
	"""Convert cost to a 0..1 score where 1 is best (<= target)."""
	if target_usd <= 0:
		return 0.0
	return _clamp01(1.0 - (float(cost_usd) / float(target_usd)))


def compute_final_score(summary: Dict[str, Any], cfg: PipelineConfig) -> Tuple[float, Dict[str, float]]:
	"""Compute weighted final score.

	final_score = 0.7*DLA + 0.2*latency_score + 0.1*cost_score (defaults)
	"""
	dla = float(summary.get("dla") or 0.0)
	lat = float(summary.get("latency_p95_sec") if cfg.leaderboard_use_latency_p95 else summary.get("latency_avg_sec") or 0.0)
	cost = float(summary.get("cost_avg_usd") or 0.0)

	lat_score = normalize_latency(lat, cfg.leaderboard_latency_target_sec)
	cost_score = normalize_cost(cost, cfg.leaderboard_cost_target_usd)

	w_dla = float(cfg.leaderboard_weight_dla)
	w_lat = float(cfg.leaderboard_weight_latency)
	w_cost = float(cfg.leaderboard_weight_cost)
	ws = w_dla + w_lat + w_cost
	if ws <= 0:
		ws = 1.0
	w_dla, w_lat, w_cost = w_dla / ws, w_lat / ws, w_cost / ws

	final = _clamp01(w_dla * _clamp01(dla) + w_lat * lat_score + w_cost * cost_score)
	components = {
		"dla": _clamp01(dla),
		"latency_score": float(lat_score),
		"cost_score": float(cost_score),
		"w_dla": float(w_dla),
		"w_latency": float(w_lat),
		"w_cost": float(w_cost),
	}
	return float(final), components
