from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

from dataclasses import replace

import time
import sys
import os

from cost_latency import estimate_cost, estimate_cost_breakdown
from explainability import save_field_overlays
from extractor import (
	aggregate_confidence,
	calibrate_doc_confidence,
	detect_signature_presence,
	detect_stamp_presence,
	extract_fields,
	extract_asset_cost,
	extract_dealer_name,
	extract_horse_power,
	extract_model_name,
	normalize_dealer_name,
)
from layout import build_structured_layout, merge_structured_layouts
from ocr_fallback import pymupdf_extract_words
from pipeline.ocr_engine import DOC_AI_FAST_MODE, OCRFailure, ocr_page_with_timeout, warmup_ocr
from postprocess import clean_int, clean_str
from utils.config import PipelineConfig
from utils.config_io import load_config_json
from utils.eda_profile import load_eda_profile, strategy_from_profile
from utils.determinism import set_deterministic
from utils.logging import get_json_logger, log_event
from utils.pdf import iter_pdf_to_images
from utils.text import detect_language_bucket
from validator import InvoiceFields, InvoiceOutput, PresenceBox
from architecture_diagram import generate_architecture_png


# Populated by CLI `main()` so wrappers can keep the requested
# `run_image_pipeline(image_path, out_path)` / `run_pdf_pipeline(pdf_path, out_path)` signatures.
_CLI_CONFIG: Optional[PipelineConfig] = None
_CLI_DOC_ID: Optional[str] = None


def _resolve_out_path(*, base_dir: Path, config: PipelineConfig, out_arg: str) -> Path:
	output_dir = config.output_dir if Path(config.output_dir).is_absolute() else (base_dir / config.output_dir)
	out_path_arg = Path(out_arg)
	if out_path_arg.is_absolute():
		out_path = out_path_arg
	else:
		# If user passed outputs/... then drop leading outputs
		parts = list(out_path_arg.parts)
		if parts and parts[0] == "outputs":
			parts = parts[1:]
		out_path = output_dir.joinpath(*parts) if parts else (output_dir / "result.json")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	return out_path


def _resolve_in_path(*, base_dir: Path, in_arg: str) -> Path:
	"""Best-effort input path resolution.

	This repo is often run from different working directories; users also
	sometimes pass paths like ../data/images/... which may not exist depending
	on CWD. We try a few common anchors (repo root, base_dir, data/images).
	"""
	p = Path(in_arg)
	if p.exists():
		return p

	# Try relative to this module directory.
	if not p.is_absolute():
		cand = base_dir / p
		if cand.exists():
			return cand

		repo_root = base_dir.parent
		cand = repo_root / p
		if cand.exists():
			return cand

		parts = list(p.parts)
		if "data" in parts:
			idx = parts.index("data")
			cand = repo_root.joinpath(*parts[idx:])
			if cand.exists():
				return cand

		# Common location: repo_root/data/images/<filename>
		cand = repo_root / "data" / "images" / p.name
		if cand.exists():
			return cand

	return p


def _write_result_json(*, result: Dict, out_path: Path) -> None:
	out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(pdf_path: str, doc_id: str, config: PipelineConfig) -> Dict:
	start_total = time.perf_counter()

	def _fallback_output(*, error: str) -> Dict:
		processing_time = float(round(time.perf_counter() - start_total, 4))
		fallback = {
			"doc_id": str(doc_id),
			"fields": {
				"dealer_name": None,
				"model_name": None,
				"horse_power": None,
				"asset_cost": None,
				"signature": {"present": False, "bbox": []},
				"stamp": {"present": False, "bbox": []},
			},
			"confidence": 0.0,
			"review_required": True,
			"processing_time_sec": processing_time,
			"cost_estimate_usd": 0.0,
			"cost_breakdown_usd": None,
			"error": str(error)[:500],
		}
		try:
			InvoiceOutput.model_validate(fallback, strict=True)
		except Exception:
			pass
		return fallback

	try:
		set_deterministic(seed=getattr(config, "seed", 1337), deterministic=getattr(config, "deterministic", True))

		base_dir = Path(__file__).resolve().parent
		output_dir = config.output_dir if Path(config.output_dir).is_absolute() else (base_dir / config.output_dir)
		overlays_dir = output_dir / "overlays"
		output_dir.mkdir(parents=True, exist_ok=True)
		overlays_dir.mkdir(parents=True, exist_ok=True)

		run_mode = str(getattr(config, "run_mode", "normal"))
		# Jury/PNG runs must print only the final JSON (no log lines on stdout).
		to_stdout = False
		logger = get_json_logger("docai", log_path=output_dir / "pipeline_logs.jsonl", to_stdout=to_stdout)

		deadline = start_total + 30.0
		log_event(logger, "pipeline_start", doc_id=doc_id, pdf_path=str(pdf_path))

		# Optional offline EDA profile (fast JSON). Never computes EDA here.
		repo_root = base_dir.parent
		profile = load_eda_profile(repo_root)
		strategy = strategy_from_profile(profile)
		log_event(
			logger,
			"eda_profile_loaded",
			header_first=bool(strategy.header_first),
			keyword_anchored=bool(strategy.keyword_anchored),
			hindi_detected=bool(strategy.hindi_detected),
			upscale_factor=float(strategy.upscale_factor),
		)

		# Determine total page count for skip accounting (cheap)
		is_digital_pdf = False
		total_pages = None
		try:
			import fitz  # PyMuPDF

			doc = fitz.open(str(pdf_path))
			total_pages = int(doc.page_count)
			try:
				probe_pages = min(2, total_pages)
				for i in range(probe_pages):
					page = doc.load_page(i)
					if (page.get_text("text") or "").strip():
						is_digital_pdf = True
						break
			except Exception:
				is_digital_pdf = False
			doc.close()
		except Exception:
			total_pages = None

		# 1) PDF -> images (latency-aware)
		t0 = time.perf_counter()
		eff_dpi = int(getattr(config, "dpi", 200))
		# Larger DPI for scanned/low-res datasets (from offline EDA)
		if float(strategy.upscale_factor) >= 1.5:
			eff_dpi = max(eff_dpi, 250)
		if float(strategy.upscale_factor) >= 2.0:
			eff_dpi = max(eff_dpi, 300)
		pages = []
		skipped_pages = 0
		deadline_epoch = time.time() + max(0.0, 29.0 - (time.perf_counter() - start_total))
		for p in iter_pdf_to_images(pdf_path, dpi=eff_dpi, max_pages=config.max_pages, deadline_epoch=deadline_epoch):
			pages.append(p)
			if time.perf_counter() >= deadline - 2.0:
				break
		page_images = [p.image for p in pages]
		if total_pages is not None:
			limit = min(total_pages, int(config.max_pages))
			skipped_pages = max(0, limit - len(page_images))
		log_event(
			logger,
			"stage_pdf_to_images",
			sec=round(time.perf_counter() - t0, 4),
			pages=len(page_images),
			dpi=eff_dpi,
			skipped_due_to_latency=skipped_pages,
		)

		# 2) OCR
		t0 = time.perf_counter()
		ocr_mode = "paddleocr"
		ocr_fallback_used = False
		ocr_failures = 0
		ocr_pages = []

		def _blank_ocr_page(page_index: int):
			return type("_", (), {"page_index": page_index, "words": [], "texts": lambda self: []})()

		allow_paddle_env = str(os.getenv("DOC_AI_ENABLE_PADDLEOCR", "0")).strip().lower() in {"1", "true", "yes"}
		# Jury-safe default: PaddleOCR is OFF unless explicitly enabled via env var.
		allow_paddle = bool(allow_paddle_env)
		use_paddle = bool((not DOC_AI_FAST_MODE) and (not is_digital_pdf) and allow_paddle)
		# Safety default: avoid PaddleOCR in evaluator-like runs unless explicitly enabled.
		if run_mode in {"submission", "judge", "demo"} and (not allow_paddle):
			use_paddle = False
			if not is_digital_pdf:
				ocr_fallback_used = True
				ocr_mode = "pymupdf_no_paddle_in_submission"
				log_event(logger, "ocr_paddle_disabled", run_mode=run_mode, reason="default_off")
		if DOC_AI_FAST_MODE:
			ocr_mode = "pymupdf_fast_mode"
			log_event(logger, "ocr_fast_mode", enabled=True)
		elif is_digital_pdf:
			ocr_mode = "pymupdf"
			log_event(logger, "ocr_engine_digital_pdf", mode=ocr_mode)
		else:
			ocr_mode = "paddleocr"

		engine_kwargs = {
			"use_angle_cls": bool(getattr(config, "use_angle_cls", True)),
			"langs": tuple(strategy.ocr_langs) if strategy.ocr_langs else tuple(getattr(config, "ocr_langs", ("en", "devanagari", "gujarati"))),
			"max_retries": int(getattr(config, "max_ocr_retries", 1)),
			"preprocess_variants": [tuple(v) for v in getattr(config, "ocr_preprocess_variants", ())] or None,
			"autorotate": bool(getattr(config, "ocr_autorotate", True)),
			"adaptive_threshold": bool(getattr(config, "ocr_adaptive_threshold", False)),
			"shadow_remove": bool(getattr(config, "ocr_shadow_remove", True)),
			"perspective_correct": bool(getattr(config, "ocr_perspective_correct", False)),
			"upscale_if_low_res": bool(getattr(config, "ocr_upscale_if_low_res", True)),
		}

		for p in pages:
			remaining = float(deadline - time.perf_counter())
			if remaining <= 2.0:
				skipped_pages += 1
				log_event(logger, "latency_guard_skip_page", stage="ocr", page_index=p.page_index)
				break

			if use_paddle and remaining < 12.0:
				use_paddle = False
				ocr_fallback_used = True
				ocr_mode = "pymupdf_latency_guard"
				log_event(logger, "ocr_latency_guard_fallback", remaining_sec=round(remaining, 3), mode=ocr_mode)

			if use_paddle:
				try:
					ocr_pages.append(
						ocr_page_with_timeout(
							page_index=p.page_index,
							image=p.image,
							seed=int(getattr(config, "seed", 1337)),
							engine_kwargs=engine_kwargs,
							timeout_sec=10.0,
						)
					)
				except OCRFailure as e:
					ocr_failures += 1
					ocr_fallback_used = True
					log_event(logger, "ocr_failure", page_index=p.page_index, error=str(e), fallback="pymupdf")
					try:
						ocr_pages.append(pymupdf_extract_words(pdf_path, p.page_index))
					except Exception:
						ocr_pages.append(_blank_ocr_page(p.page_index))
				except Exception as e:
					ocr_failures += 1
					log_event(logger, "ocr_failure", page_index=p.page_index, error=str(e))
					ocr_pages.append(_blank_ocr_page(p.page_index))
			else:
				try:
					ocr_pages.append(pymupdf_extract_words(pdf_path, p.page_index))
				except Exception:
					ocr_pages.append(_blank_ocr_page(p.page_index))

		ocr_sec = time.perf_counter() - t0
		avg_ocr_conf = 0.0
		word_counts = [len(getattr(op, "words", [])) for op in ocr_pages]
		if sum(word_counts) > 0:
			confs = [w.conf for op in ocr_pages for w in getattr(op, "words", [])]
			avg_ocr_conf = float(sum(confs) / max(1, len(confs))) if confs else 0.0
		log_event(
			logger,
			"stage_ocr",
			sec=round(ocr_sec, 4),
			mode=ocr_mode,
			avg_word_conf=round(avg_ocr_conf, 4),
			word_counts=word_counts,
			failures=ocr_failures,
		)

		# 3) Layout (structured regions)
		t0 = time.perf_counter()
		structured_pages = []
		for op, img in zip(ocr_pages, page_images):
			if time.perf_counter() >= deadline - 1.5:
				log_event(logger, "latency_guard_skip_page", stage="layout", page_index=getattr(op, "page_index", -1))
				break
			structured_pages.append(build_structured_layout(op, (img.width, img.height)))
		doc_layout = merge_structured_layouts(structured_pages)
		language = detect_language_bucket([ln.text for ln in doc_layout.all_lines])
		layout_sec = time.perf_counter() - t0
		region_counts = {k: len(doc_layout.regions.get(k).lines) for k in ("header", "body", "table", "footer") if k in doc_layout.regions}
		log_event(logger, "stage_layout", sec=round(layout_sec, 4), language=language, region_line_counts=region_counts)

		# 4) Extraction (region-aware)
		region_overrides = dict(getattr(config, "region_weight_overrides", {}) or {})
		if strategy.header_first:
			region_overrides.setdefault("table", {})
			region_overrides["table"].setdefault("header", 0.95)
			region_overrides.setdefault("footer", {})
			region_overrides["footer"].setdefault("header", 0.85)

		t0 = time.perf_counter()
		ocr_text = "\n".join([ln.text for ln in doc_layout.all_lines if (ln.text or "").strip()])
		dealer = extract_dealer_name(
			doc_layout,
			base_dir=base_dir,
			threshold=config.dealer_fuzzy_threshold,
			region_weight_overrides=region_overrides,
			keyword_anchored=bool(strategy.keyword_anchored),
		)
		model = extract_model_name(
			doc_layout,
			base_dir=base_dir,
			region_weight_overrides=region_overrides,
			keyword_anchored=bool(strategy.keyword_anchored),
		)
		hp = extract_horse_power(doc_layout, region_weight_overrides=region_overrides)
		asset = extract_asset_cost(doc_layout, region_weight_overrides=region_overrides, header_first=bool(strategy.header_first))
		per_field = {"dealer_name": dealer, "model_name": model, "horse_power": hp, "asset_cost": asset}
		base_conf = aggregate_confidence(per_field)
		doc_conf = calibrate_doc_confidence(
			per_field=per_field,
			layout=doc_layout,
			base_conf=base_conf,
			ocr_fallback_used=bool(ocr_fallback_used),
			run_mode=str(getattr(config, "run_mode", "normal")),
		)
		extract_sec = time.perf_counter() - t0
		log_event(
			logger,
			"stage_extraction",
			sec=round(extract_sec, 4),
			fields={k: v.as_log_dict() for k, v in per_field.items()},
			doc_conf=float(doc_conf),
		)

		# 5) Vision detection
		# Jury-safe: disable signature/stamp detection entirely (no YOLO init, no model files).
		detector = None
		sig = None
		stamp = None
		log_event(logger, "stage_vision", sec=0.0, yolo_used=False, signature_present=False, stamp_present=False)

		# 6) Assemble + validate
		t0 = time.perf_counter()
		dealer_out = normalize_dealer_name(clean_str(dealer.value if isinstance(dealer.value, str) else None))
		signature_present = bool(detect_signature_presence(ocr_text))
		stamp_present = bool(detect_stamp_presence(ocr_text))
		fields = InvoiceFields(
			dealer_name=dealer_out,
			model_name=clean_str(model.value if isinstance(model.value, str) else None),
			horse_power=clean_int(hp.value),
			asset_cost=clean_int(asset.value),
			signature=PresenceBox(present=signature_present, bbox=sig.bbox.as_list() if sig else []),
			stamp=PresenceBox(present=stamp_present, bbox=stamp.bbox.as_list() if stamp else []),
		)

		pages_n = len(page_images)
		yolo_used = False
		cost_breakdown = estimate_cost_breakdown(config, pages=pages_n, yolo_used=False)
		cost = float(cost_breakdown.get("total", estimate_cost(config, pages=pages_n, yolo_used=False)))

		processing_time = float(round(time.perf_counter() - start_total, 3))
		validated = InvoiceOutput(
			doc_id=doc_id,
			fields=fields,
			confidence=float(doc_conf),
			review_required=bool(float(doc_conf) < float(getattr(config, "review_conf_threshold", 0.75))),
			processing_time_sec=processing_time,
			cost_estimate_usd=cost,
			cost_breakdown_usd=cost_breakdown,
		)
		validate_sec = time.perf_counter() - t0
		latency_ok = processing_time <= 30.0
		cost_ok = cost <= 0.01
		log_event(
			logger,
			"stage_validation",
			sec=round(validate_sec, 4),
			latency_ok=latency_ok,
			cost_ok=cost_ok,
			processing_time_sec=processing_time,
			cost_estimate_usd=cost,
			cost_breakdown_usd=cost_breakdown,
		)

		# 7) Explainability overlays
		if config.save_overlays and page_images:
			field_boxes = {
				"dealer_name": dealer.bbox,
				"model_name": model.bbox,
				"horse_power": hp.bbox,
				"asset_cost": asset.bbox,
			}
			save_field_overlays(
				doc_id=doc_id,
				page_images=page_images[:1],
				field_boxes=field_boxes,
				field_confs={
					"dealer_name": float(dealer.conf),
					"model_name": float(model.conf),
					"horse_power": float(hp.conf),
					"asset_cost": float(asset.conf),
				},
				sig_box=sig.bbox if sig else None,
				stamp_box=stamp.bbox if stamp else None,
				output_dir=overlays_dir,
			)
			log_event(logger, "stage_explainability", overlays_dir=str(overlays_dir))

		run_log = {
			"doc_id": doc_id,
			"language": language,
			"confidence": float(doc_conf),
			"run_mode": str(getattr(config, "run_mode", "normal")),
			"processing_time_sec": processing_time,
			"cost_estimate_usd": cost,
			"cost_ok": bool(cost_ok),
			"cost_breakdown_usd": cost_breakdown,
			"error_flag": bool(doc_conf < 0.75) or (not latency_ok),
			"latency_ok": bool(latency_ok),
			"skipped_pages": int(skipped_pages),
			"ocr_failures": int(ocr_failures),
			"ocr_mode": ocr_mode,
			"ocr_fallback_used": bool(ocr_fallback_used),
			"avg_ocr_conf": round(avg_ocr_conf, 4),
			"region_line_counts": region_counts,
			"yolo_used": False,
			"signature_present": bool(signature_present),
			"stamp_present": bool(stamp_present),
			"fields": {k: v.as_log_dict() for k, v in per_field.items()},
		}
		log_path = output_dir / "runs.jsonl"
		with log_path.open("a", encoding="utf-8") as f:
			f.write(json.dumps(run_log, ensure_ascii=False) + "\n")

		log_event(logger, "pipeline_end", doc_id=doc_id, processing_time_sec=processing_time, latency_ok=latency_ok)
		return validated.model_dump()
	except BaseException as e:
		# Keep submission/judge/demo stdout+stderr clean; logs are already persisted to jsonl.
		try:
			run_mode = str(getattr(config, "run_mode", "normal"))
			if run_mode not in {"submission", "judge", "demo"}:
				print(f"[warn] pipeline failed for {doc_id}: {e}", file=sys.stderr)
		except Exception:
			pass
		return _fallback_output(error=str(e))


def _run_image_pipeline_core(image_path: str, doc_id: str, config: PipelineConfig) -> Dict:
	"""PNG-only pipeline entrypoint for jury evaluation.

	- Always treats input as a single-page scanned document
	- Uses offline EDA profile (if present) to choose strategy knobs
	- Never runs EDA during extraction
	"""
	start_total = time.perf_counter()

	def _fallback_output(*, error: str) -> Dict:
		processing_time = float(round(time.perf_counter() - start_total, 4))
		fallback = {
			"doc_id": str(doc_id),
			"fields": {
				"dealer_name": None,
				"model_name": None,
				"horse_power": None,
				"asset_cost": None,
				"signature": {"present": False, "bbox": []},
				"stamp": {"present": False, "bbox": []},
			},
			"confidence": 0.0,
			"review_required": True,
			"processing_time_sec": processing_time,
			"cost_estimate_usd": 0.0,
			"cost_breakdown_usd": None,
			"error": str(error)[:500],
		}
		try:
			InvoiceOutput.model_validate(fallback, strict=True)
		except Exception:
			pass
		return fallback

	try:
		set_deterministic(seed=getattr(config, "seed", 1337), deterministic=getattr(config, "deterministic", True))

		base_dir = Path(__file__).resolve().parent
		repo_root = base_dir.parent
		output_dir = config.output_dir if Path(config.output_dir).is_absolute() else (base_dir / config.output_dir)
		overlays_dir = output_dir / "overlays"
		output_dir.mkdir(parents=True, exist_ok=True)
		overlays_dir.mkdir(parents=True, exist_ok=True)

		run_mode = str(getattr(config, "run_mode", "normal"))
		# Jury/PNG runs: keep stdout clean (final JSON only).
		to_stdout = False
		logger = get_json_logger("docai", log_path=output_dir / "pipeline_logs.jsonl", to_stdout=to_stdout)

		deadline = start_total + 30.0
		log_event(logger, "pipeline_start", doc_id=doc_id, image_path=str(image_path))

		# Load optional offline EDA profile (fast JSON)
		profile = load_eda_profile(repo_root)
		strategy = strategy_from_profile(profile)
		log_event(
			logger,
			"eda_profile_loaded",
			header_first=bool(strategy.header_first),
			keyword_anchored=bool(strategy.keyword_anchored),
			hindi_detected=bool(strategy.hindi_detected),
			upscale_factor=float(strategy.upscale_factor),
		)

		# 1) PNG -> image
		t0 = time.perf_counter()
		from PIL import Image

		img = Image.open(str(image_path)).convert("RGB")
		if float(strategy.upscale_factor) > 1.01:
			w = int(round(img.width * float(strategy.upscale_factor)))
			h = int(round(img.height * float(strategy.upscale_factor)))
			img = img.resize((w, h), resample=Image.Resampling.LANCZOS)
		log_event(logger, "stage_image_load", sec=round(time.perf_counter() - t0, 4), size=[img.width, img.height])

		# 2) OCR (PaddleOCR worker)
		t0 = time.perf_counter()
		ocr_mode = "paddleocr"
		ocr_fallback_used = False
		ocr_failures = 0
		ocr_pages = []

		engine_kwargs = {
			"use_angle_cls": bool(getattr(config, "use_angle_cls", True)),
			"langs": tuple(strategy.ocr_langs) if strategy.ocr_langs else tuple(getattr(config, "ocr_langs", ("en", "devanagari", "gujarati"))),
			"max_retries": int(getattr(config, "max_ocr_retries", 1)),
			"preprocess_variants": [tuple(v) for v in getattr(config, "ocr_preprocess_variants", ())] or None,
			"autorotate": bool(getattr(config, "ocr_autorotate", True)),
			"adaptive_threshold": bool(getattr(config, "ocr_adaptive_threshold", False)),
			"shadow_remove": bool(getattr(config, "ocr_shadow_remove", True)),
			"perspective_correct": bool(getattr(config, "ocr_perspective_correct", False)),
			"upscale_if_low_res": bool(getattr(config, "ocr_upscale_if_low_res", True)),
		}

		# PNG runs: PaddleOCR is enabled only if explicitly allowed (jury-safe).
		# Requirement: ensure PaddleOCR is used when DOC_AI_ENABLE_PADDLEOCR=1.
		allow_paddle_env = str(os.getenv("DOC_AI_ENABLE_PADDLEOCR", "0")).strip().lower() in {"1", "true", "yes"}
		allow_paddle_cfg = bool(getattr(config, "enable_paddleocr", False))
		disable_paddle_env = str(os.getenv("DOC_AI_DISABLE_PADDLEOCR", "0")).strip().lower() in {"1", "true", "yes"}
		use_paddle = bool((not DOC_AI_FAST_MODE) and (allow_paddle_env or allow_paddle_cfg) and (not disable_paddle_env))
		warmed = False
		if DOC_AI_FAST_MODE:
			ocr_mode = "fast_mode_no_paddle"
		elif not use_paddle:
			ocr_mode = "no_paddle"
		else:
			ocr_mode = "paddleocr"
			# Warm models once; if warmup fails, worker will return empty OCR and we'll fallback.
			remaining = float(deadline - time.perf_counter())
			warmed = warmup_ocr(
				seed=int(getattr(config, "seed", 1337)),
				engine_kwargs=engine_kwargs,
				timeout_sec=min(8.0, max(1.5, remaining - 2.0)),
			)
			log_event(logger, "ocr_warmup", ok=bool(warmed))

		# Primary: PaddleOCR
		page = None
		if use_paddle:
			try:
				remaining = float(deadline - time.perf_counter())
				page = ocr_page_with_timeout(
					page_index=0,
					image=img,
					seed=int(getattr(config, "seed", 1337)),
					engine_kwargs=engine_kwargs,
					timeout_sec=min(10.0, max(2.0, remaining - 2.0)),
				)
			except OCRFailure as e:
				ocr_failures += 1
				ocr_fallback_used = True
				ocr_mode = "paddleocr_failed"
				log_event(logger, "ocr_failure", page_index=0, error=str(e), fallback="tesseract")
				page = None
			except Exception as e:
				ocr_failures += 1
				ocr_fallback_used = True
				ocr_mode = "paddleocr_failed"
				log_event(logger, "ocr_failure", page_index=0, error=str(e), fallback="tesseract")
				page = None

		# Fallback: Tesseract (if installed)
		if (page is None) or (not getattr(page, "words", [])):
			try:
				from ocr import tesseract_run_page
				# Prefer fast/stable English-only first.
				# Using multiple languages can significantly slow Tesseract.
				fallback_langs = ["en"]

				def _score_candidate(*, cand_fields: Dict, cand_text: str, word_count: int) -> int:
					m = (cand_fields or {}).get("model_name")
					c = (cand_fields or {}).get("asset_cost")
					# Penalize obviously bad model/cost to steer selection.
					bad_model = bool(
						re.search(
							r"(?i)\b(authori[sz]ed\s+dealer|authori[sz]ed\s+distributor|dealer\b|implements?\b|spares?\b|parts?\b)\b",
							str(m or ""),
						)
						or re.search(r"(?i)\b(servicing|service|free\s+servic|materials|terms|conditions|delivery)\b", str(m or ""))
						or re.search(r"(?i)\b(hood|hitch|trailer|trailor|plough|tiller|leveller|cage\s*wheel)\b", str(m or ""))
					)
					# Contact/ID lines (ph/gst/tin) often create false asset costs.
					bad_cost = False
					if c is not None:
						try:
							ci = int(c)
							bad_cost = ci < 10000 or bool(
								re.search(
									rf"(?i)\b(ph\.?|phone|mob\.?|mobile|tel\.?|tin\b|gst\b|pan\b|a/c\b|ac\b|ifsc|bank)\b[^\n]*\b{ci}\b",
									cand_text,
								)
							)
						except Exception:
							bad_cost = True

					dealer_ok = bool((cand_fields or {}).get("dealer_name"))
					model_ok = bool((cand_fields or {}).get("model_name")) and (not bad_model)
					hp_ok = bool((cand_fields or {}).get("horse_power"))
					cost_ok = bool((cand_fields or {}).get("asset_cost")) and (not bad_cost)
					# Prioritize numeric fields (HP/cost) over dealer/model.
					filled_score = (3 * int(hp_ok) + 3 * int(cost_ok) + 1 * int(dealer_ok) + 1 * int(model_ok))
					return int(filled_score * 1000 + int(word_count))

				# Try a small set of PSM modes and pick the one that yields the
				# best downstream field extraction (not necessarily the most words).
				best = None
				best_psm = None
				best_score = -1
				best_fields: Dict = {}
				remaining = float(deadline - time.perf_counter())
				# Reserve time for potential mixed-language OCR + layout/extraction.
				per_try = min(4.5, max(2.5, remaining - 12.0))
				# Pass 1: English-only
				for psm in (6, 3):
					if float(deadline - time.perf_counter()) <= 2.0:
						break
					cand = tesseract_run_page(0, img, langs=fallback_langs, psm=int(psm), timeout_sec=float(per_try))
					if not getattr(cand, "words", []):
						continue
					cand_layout = build_structured_layout(cand, (img.width, img.height))
					cand_text = "\n".join([ln.text for ln in cand_layout.all_lines if (ln.text or "").strip()])
					cand_fields = extract_fields(cand_text) if cand_text.strip() else {}
					score = _score_candidate(
						cand_fields=cand_fields,
						cand_text=cand_text,
						word_count=len(getattr(cand, "words", [])),
					)
					if score > best_score:
						best_score = score
						best = cand
						best_psm = int(psm)
						best_fields = dict(cand_fields or {})
					# Early exit if we got a strong result.
					if best_score >= 6000:
						break

				# Pass 2 (conditional): try mixed Hindi+English if numeric fields are still missing
				need_numeric = (not bool((best_fields or {}).get("horse_power"))) or (not bool((best_fields or {}).get("asset_cost")))
				# Only attempt the slower mixed-language OCR when we have enough budget left.
				if need_numeric and float(deadline - time.perf_counter()) > 10.0:
					mixed_langs = ["en", "devanagari"]
					for psm in (3,):
						if float(deadline - time.perf_counter()) <= 2.0:
							break
						cand_timeout = float(min(8.0, max(4.0, deadline - time.perf_counter() - 5.0)))
						cand = tesseract_run_page(0, img, langs=mixed_langs, psm=int(psm), timeout_sec=cand_timeout)
						if not getattr(cand, "words", []):
							continue
						cand_layout = build_structured_layout(cand, (img.width, img.height))
						cand_text = "\n".join([ln.text for ln in cand_layout.all_lines if (ln.text or "").strip()])
						cand_fields = extract_fields(cand_text) if cand_text.strip() else {}
						score = _score_candidate(
							cand_fields=cand_fields,
							cand_text=cand_text,
							word_count=len(getattr(cand, "words", [])),
						)
						if score > best_score:
							best_score = score
							best = cand
							best_psm = int(psm)
							best_fields = dict(cand_fields or {})

				if best is not None:
					page = best
					ocr_fallback_used = True
					ocr_mode = f"tesseract_psm{best_psm}" if best_psm is not None else "tesseract"
				else:
					# Fully offline-safe: return empty OCR (still schema-valid output).
					page = type("_", (), {"page_index": 0, "words": [], "texts": lambda self: []})()
					if not use_paddle:
						ocr_mode = "empty_no_paddle"
					else:
						ocr_mode = "empty_after_fallback"
			except Exception as e:
				ocr_fallback_used = True
				log_event(logger, "ocr_fallback_unavailable", error=str(e)[:500])
				page = type("_", (), {"page_index": 0, "words": [], "texts": lambda self: []})()
				ocr_mode = "empty_fallback_unavailable"

		ocr_pages.append(page)
		ocr_sec = time.perf_counter() - t0
		word_counts = [len(getattr(op, "words", [])) for op in ocr_pages]
		avg_ocr_conf = 0.0
		if sum(word_counts) > 0:
			confs = [w.conf for op in ocr_pages for w in getattr(op, "words", [])]
			avg_ocr_conf = float(sum(confs) / max(1, len(confs))) if confs else 0.0
		log_event(
			logger,
			"stage_ocr",
			sec=round(ocr_sec, 4),
			mode=ocr_mode,
			avg_word_conf=round(avg_ocr_conf, 4),
			word_counts=word_counts,
			failures=ocr_failures,
			paddle_enabled=bool(use_paddle),
			paddle_warmed=bool(warmed),
		)

		# 3) Layout
		t0 = time.perf_counter()
		structured_pages = [build_structured_layout(ocr_pages[0], (img.width, img.height))]
		doc_layout = merge_structured_layouts(structured_pages)
		language = detect_language_bucket([ln.text for ln in doc_layout.all_lines])
		region_counts = {k: len(doc_layout.regions.get(k).lines) for k in ("header", "body", "table", "footer") if k in doc_layout.regions}
		log_event(logger, "stage_layout", sec=round(time.perf_counter() - t0, 4), language=language, region_line_counts=region_counts)

		# 4) Extraction (EDA-driven strategy)
		# Region weight tuning for header-first docs
		region_overrides = dict(getattr(config, "region_weight_overrides", {}) or {})
		if strategy.header_first:
			region_overrides.setdefault("table", {})
			region_overrides["table"].setdefault("header", 0.95)
			region_overrides.setdefault("footer", {})
			region_overrides["footer"].setdefault("header", 0.85)

		t0 = time.perf_counter()
		ocr_text = "\n".join([ln.text for ln in doc_layout.all_lines if (ln.text or "").strip()])
		dealer = extract_dealer_name(
			doc_layout,
			base_dir=base_dir,
			threshold=config.dealer_fuzzy_threshold,
			region_weight_overrides=region_overrides,
			keyword_anchored=bool(strategy.keyword_anchored),
		)
		model = extract_model_name(
			doc_layout,
			base_dir=base_dir,
			region_weight_overrides=region_overrides,
			keyword_anchored=bool(strategy.keyword_anchored),
		)
		hp = extract_horse_power(doc_layout, region_weight_overrides=region_overrides)
		asset = extract_asset_cost(doc_layout, region_weight_overrides=region_overrides, header_first=bool(strategy.header_first))
		per_field = {"dealer_name": dealer, "model_name": model, "horse_power": hp, "asset_cost": asset}

		# Fallback: if OCR produced text but structured extraction missed, use
		# simple regex rules (per prompt) to populate missing fields.
		fallback = extract_fields(ocr_text)
		fallback_dealer = fallback.get("dealer_name") if isinstance(fallback, dict) else None
		fallback_model = fallback.get("model_name") if isinstance(fallback, dict) else None
		fallback_hp = fallback.get("horse_power") if isinstance(fallback, dict) else None
		fallback_cost = fallback.get("asset_cost") if isinstance(fallback, dict) else None

		def _bad_model_text(s: str | None) -> bool:
			if not s:
				return True
			ss = str(s)
			return bool(
				re.search(
					r"(?i)\b(servicing|service|free\s+servic|materials|guarantee|terms|conditions|subsidy|permit|delivery)\b",
					ss,
				)
				or re.search(r"(?i)\b(hood|hitch|trailer|trailor|plough|tiller|leveller|cage\s*wheel|toolskit|battery)\b", ss)
			)

		def _looks_like_contact_number(v: int | None, text: str) -> bool:
			if v is None:
				return False
			pat = re.compile(
				rf"(?i)\b(ph\.?|phone|mob\.?|mobile|mo\.?|tel\.?|tin\b|gst\b|pan\b|a/c\b|ac\b|account|ifsc|bank)\b[^\n]*\b{int(v)}\b"
			)
			return bool(pat.search(text or ""))

		base_conf = aggregate_confidence(per_field)
		doc_conf = calibrate_doc_confidence(
			per_field=per_field,
			layout=doc_layout,
			base_conf=base_conf,
			ocr_fallback_used=bool(ocr_fallback_used),
			run_mode=run_mode,
		)

		# Vision disabled for jury runs.
		detector = None
		sig = None
		stamp = None
		log_event(logger, "stage_detection", sec=0.0, yolo_used=False)

		processing_time = float(round(time.perf_counter() - start_total, 4))
		cost_breakdown = estimate_cost_breakdown(config=config, pages=1, yolo_used=False)
		cost_estimate = estimate_cost(config=config, pages=1, yolo_used=False)

		chosen_model = clean_str(model.value) if model.value is not None else None
		if _bad_model_text(chosen_model):
			chosen_model = clean_str(fallback_model)

		fallback_cost_i = clean_int(fallback_cost)
		chosen_cost = clean_int(asset.value) if asset.value is not None else fallback_cost_i
		# Replace suspicious contact/ID-derived numbers with fallback.
		if chosen_cost is not None and _looks_like_contact_number(int(chosen_cost), ocr_text):
			chosen_cost = fallback_cost_i
		# Enforce a minimal plausible rupee amount; if structured is tiny, try fallback.
		if chosen_cost is not None and int(chosen_cost) < 10000:
			chosen_cost = fallback_cost_i
			if chosen_cost is not None and int(chosen_cost) < 10000:
				chosen_cost = None

		chosen_hp = clean_int(hp.value) if hp.value is not None else clean_int(fallback_hp)
		if chosen_hp is not None and not (10 <= int(chosen_hp) <= 125):
			chosen_hp = clean_int(fallback_hp)
			if chosen_hp is not None and not (10 <= int(chosen_hp) <= 125):
				chosen_hp = None

		out = {
			"doc_id": str(doc_id),
			"fields": {
				"dealer_name": normalize_dealer_name(
					clean_str(dealer.value) if dealer.value is not None else clean_str(fallback_dealer)
				),
				"model_name": chosen_model,
				"horse_power": chosen_hp,
				"asset_cost": chosen_cost,
				"signature": {"present": bool(detect_signature_presence(ocr_text)), "bbox": []},
				"stamp": {"present": bool(detect_stamp_presence(ocr_text)), "bbox": []},
			},
			"confidence": float(doc_conf),
			"review_required": bool(doc_conf < float(getattr(config, "review_conf_threshold", 0.75))),
			"processing_time_sec": float(processing_time),
			"cost_estimate_usd": float(cost_estimate),
			"cost_breakdown_usd": cost_breakdown,
		}

		validated = InvoiceOutput.model_validate(out, strict=True)
		# Lightweight run log (still deterministic, no EDA)
		run_log = {
			"doc_id": str(doc_id),
			"input": "png",
			"image_path": str(image_path),
			"language": str(language),
			"processing_time_sec": float(processing_time),
			"avg_ocr_conf": round(avg_ocr_conf, 4),
			"ocr_mode": ocr_mode,
			"ocr_failures": int(ocr_failures),
			"fields": {k: v.as_log_dict() for k, v in per_field.items()},
		}
		log_path = output_dir / "runs.jsonl"
		with log_path.open("a", encoding="utf-8") as f:
			f.write(json.dumps(run_log, ensure_ascii=False) + "\n")

		log_event(logger, "pipeline_end", doc_id=doc_id, processing_time_sec=processing_time)
		return validated.model_dump()
	except BaseException as e:
		try:
			if str(getattr(config, "run_mode", "normal")) not in {"submission", "judge", "demo"}:
				print(f"[warn] pipeline failed for {doc_id}: {e}", file=sys.stderr)
		except Exception:
			pass
		return _fallback_output(error=str(e))


def run_image_pipeline(image_path: str, out_path: str) -> Dict:
	"""PNG → OCR → Layout → Extraction → Validation → JSON.

	This wrapper exists to support jury-style execution with `--png`.
	"""
	base_dir = Path(__file__).resolve().parent
	config = _CLI_CONFIG or PipelineConfig()
	doc_id = _CLI_DOC_ID or Path(image_path).stem
	result = _run_image_pipeline_core(image_path=image_path, doc_id=doc_id, config=config)
	final_out_path = _resolve_out_path(base_dir=base_dir, config=config, out_arg=out_path)
	_write_result_json(result=result, out_path=final_out_path)
	return result


def run_pdf_pipeline(pdf_path: str, out_path: str) -> Dict:
	"""PDF pipeline wrapper that preserves the existing PDF flow."""
	base_dir = Path(__file__).resolve().parent
	config = _CLI_CONFIG or PipelineConfig()
	doc_id = _CLI_DOC_ID or Path(pdf_path).stem
	result = run_pipeline(pdf_path=pdf_path, doc_id=doc_id, config=config)
	final_out_path = _resolve_out_path(base_dir=base_dir, config=config, out_arg=out_path)
	_write_result_json(result=result, out_path=final_out_path)
	return result


def main() -> int:
	p = argparse.ArgumentParser(description="DocAI Invoice Extractor")
	g = p.add_mutually_exclusive_group(required=True)
	g.add_argument("--png", default="", help="Path to invoice PNG (jury final input)")
	g.add_argument("--pdf", default="", help="Path to invoice PDF (legacy)")
	p.add_argument("--doc-id", default=None, help="Doc id for output JSON")
	p.add_argument(
		"--config",
		default="",
		help="Optional JSON config (e.g. best_config.json). CLI flags override it.",
	)
	p.add_argument("--dpi", type=int, default=200)
	p.add_argument("--max-pages", type=int, default=5)
	p.add_argument(
		"--enable-paddleocr",
		action="store_true",
		help="Enable PaddleOCR (slower, higher accuracy)",
	)
	p.add_argument("--out", default="outputs/result.json", help="Output JSON path (defaults under outputs/)")
	# IMPORTANT: Offline EDA must not run during extraction. These are opt-in.
	p.add_argument("--diagram", action="store_true", help="(Optional) Generate architecture diagram")
	p.add_argument("--eda-artifacts", action="store_true", help="(Optional) Generate run-log EDA artifacts")
	p.add_argument("--error-report", action="store_true", help="(Optional) Generate error report")
	args = p.parse_args()

	if bool(getattr(args, "enable_paddleocr", False)):
		os.environ["DOC_AI_ENABLE_PADDLEOCR"] = "1"

	base_dir = Path(__file__).resolve().parent
	input_path_raw = args.png or args.pdf

	# User-friendly recovery: sometimes people accidentally do
	#   --png path/to/img.png/outputs/result.json
	# instead of
	#   --png path/to/img.png --out outputs/result.json
	# Try to auto-split when we see both an image/pdf extension and a .json.
	if args.png and (".png" in str(args.png).lower() or ".jpg" in str(args.png).lower() or ".jpeg" in str(args.png).lower()):
		s = str(args.png)
		lower = s.lower()
		if ".json" in lower and (".png" in lower or ".jpg" in lower or ".jpeg" in lower):
			# Split at the first image extension occurrence.
			for ext in (".png", ".jpg", ".jpeg"):
				idx = lower.find(ext)
				if idx != -1:
					cut = idx + len(ext)
					img_part = s[:cut]
					out_part = s[cut:]
					# If the remainder looks like an output path, adopt it.
					if ".json" in out_part.lower():
						args.png = img_part
						# Trim leading separators so it behaves like a normal relative path.
						out_part = out_part.lstrip("/\\")
						if out_part:
							args.out = out_part
					break

	resolved_input = _resolve_in_path(base_dir=base_dir, in_arg=str(args.png or args.pdf))
	if not resolved_input.exists():
		print(f"[error] input not found: {resolved_input}", file=sys.stderr)
		if args.png:
			print(
				"[hint] usage: python executable.py --png <image.png> --out outputs/result.json",
				file=sys.stderr,
			)
		else:
			print(
				"[hint] usage: python executable.py --pdf <file.pdf> --out outputs/result.json",
				file=sys.stderr,
			)
		return 2

	doc_id = args.doc_id or resolved_input.stem
	if args.config:
		cfg_path = Path(args.config)
		if not cfg_path.is_absolute():
			cfg_path = base_dir / cfg_path
		cfg = load_config_json(cfg_path)
	else:
		cfg = PipelineConfig()

	# CLI overrides (useful for ad-hoc runs even in submission mode)
	cfg = replace(
		cfg,
		dpi=int(args.dpi),
		max_pages=int(args.max_pages),
		enable_paddleocr=bool(getattr(args, "enable_paddleocr", False)),
	)

	# Expose CLI-derived context to the required wrapper signatures.
	global _CLI_CONFIG, _CLI_DOC_ID
	_CLI_CONFIG = cfg
	_CLI_DOC_ID = doc_id

	if args.png:
		result = run_image_pipeline(str(resolved_input), args.out)
	else:
		result = run_pdf_pipeline(str(resolved_input), args.out)

	output_dir = cfg.output_dir if Path(cfg.output_dir).is_absolute() else (base_dir / cfg.output_dir)

	# Optional artifacts (explicit opt-in only)
	try:
		if args.diagram:
			generate_architecture_png(output_dir / "architecture_diagram.png")
	except Exception as e:
		print(f"[warn] diagram generation failed: {e}", file=sys.stderr)
	try:
		if args.eda_artifacts:
			from eda import run_eda  # local module (doc_ai_project/eda.py)

			run_eda(runs_jsonl=output_dir / "runs.jsonl", outputs_dir=output_dir)
	except Exception as e:
		print(f"[warn] EDA generation failed: {e}", file=sys.stderr)
	try:
		if args.error_report:
			from error_analysis import run_error_analysis

			run_error_analysis(runs_jsonl=output_dir / "runs.jsonl", outputs_dir=output_dir)
	except Exception as e:
		print(f"[warn] error analysis failed: {e}", file=sys.stderr)

	print(json.dumps(result, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
