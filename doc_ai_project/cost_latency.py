from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from utils.config import PipelineConfig


@dataclass(frozen=True)
class CostLatency:
	pages: int
	processing_time_sec: float
	cost_estimate_usd: float
	cost_breakdown_usd: Dict[str, float]


def estimate_cost(config: PipelineConfig, pages: int, yolo_used: bool) -> float:
	cost = config.cost_per_doc_usd_overhead + pages * config.cost_per_page_usd_ocr
	if yolo_used:
		cost += pages * config.cost_per_page_usd_yolo
	return float(round(cost, 6))


def estimate_cost_breakdown(config: PipelineConfig, pages: int, yolo_used: bool) -> Dict[str, float]:
	"""Hackathon-friendly CPU cost estimator.

	This is a heuristic *budgeting* model: tune per-stage constants in PipelineConfig.
	"""
	overhead = float(config.cost_per_doc_usd_overhead)
	ocr = float(pages * config.cost_per_page_usd_ocr)
	vision = float(pages * config.cost_per_page_usd_yolo) if yolo_used else 0.0
	# Lightweight stages
	layout = float(0.00005 * pages)
	extraction = float(0.00003)
	validation = float(0.00001)

	breakdown = {
		"overhead": round(overhead, 6),
		"ocr": round(ocr, 6),
		"layout": round(layout, 6),
		"extraction": round(extraction, 6),
		"vision": round(vision, 6),
		"validation": round(validation, 6),
	}
	breakdown["total"] = round(sum(breakdown.values()), 6)
	return breakdown
