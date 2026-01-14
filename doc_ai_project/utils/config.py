from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class PipelineConfig:
	# Reproducibility
	seed: int = 1337
	deterministic: bool = True
	# Run context (used for confidence calibration)
	run_mode: str = "normal"  # normal|replay|tuning|submission

	# IO
	dpi: int = 200
	max_pages: int = 5
	output_dir: Path = Path("outputs")
	eda_dir: Path = Path("eda_outputs")

	# OCR
	ocr_langs: tuple[str, ...] = ("en", "devanagari", "gujarati")
	use_angle_cls: bool = True
	ocr_det: bool = True
	ocr_rec: bool = True
	max_ocr_retries: int = 1
	# Preprocessing variants to try (denoise, deskew, contrast)
	ocr_preprocess_variants: Tuple[Tuple[bool, bool, bool], ...] = (
		(True, True, True),
		(False, True, True),
		(True, False, True),
	)
	# Robustness (OpenCV-only features; ignored if cv2 missing)
	ocr_autorotate: bool = True
	ocr_adaptive_threshold: bool = False
	ocr_perspective_correct: bool = False
	ocr_shadow_remove: bool = True
	ocr_upscale_if_low_res: bool = True

	# Safe-mode
	review_conf_threshold: float = 0.75

	# Matching thresholds
	dealer_fuzzy_threshold: int = 90
	# Region-weight overrides for confidence calibration.
	# Format: {expected_region: {actual_region: weight}}
	region_weight_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)

	# Signature/stamp detection
	yolo_weights_path: Path | None = None  # set to a .pt file to enable YOLOv8
	yolo_conf: float = 0.25
	# NMS IoU threshold for post-processing
	yolo_iou: float = 0.5
	# Multi-scale inference (img sizes passed to YOLO predict)
	yolo_img_sizes: Tuple[int, ...] = (640,)

	# Hackathon leaderboard scoring
	leaderboard_weight_dla: float = 0.7
	leaderboard_weight_latency: float = 0.2
	leaderboard_weight_cost: float = 0.1
	leaderboard_latency_target_sec: float = 30.0
	leaderboard_cost_target_usd: float = 0.01
	leaderboard_use_latency_p95: bool = True

	# Submission constraints
	submission_min_dla: float = 0.95
	submission_max_latency_sec: float = 30.0
	submission_max_cost_usd: float = 0.01

	# Explainability outputs
	save_overlays: bool = True

	# Cost model (rough, hackathon-friendly)
	cost_per_page_usd_ocr: float = 0.0006
	cost_per_page_usd_yolo: float = 0.0003
	cost_per_doc_usd_overhead: float = 0.0002
