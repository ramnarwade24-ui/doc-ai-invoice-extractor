from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from dataclasses import replace

import time
import sys
import os

from cost_latency import estimate_cost, estimate_cost_breakdown
from detector import SignatureStampDetector, best_by_label
from eda import run_eda
from error_analysis import run_error_analysis
from explainability import save_field_overlays
from extractor import (
	aggregate_confidence,
	calibrate_doc_confidence,
	extract_asset_cost,
	extract_dealer_name,
	extract_horse_power,
	extract_model_name,
)
from layout import build_structured_layout, merge_structured_layouts
from ocr_fallback import pymupdf_extract_words
from pipeline.ocr_engine import DOC_AI_FAST_MODE, OCRFailure, ocr_page_with_timeout
from postprocess import clean_int, clean_str
from utils.config import PipelineConfig
from utils.config_io import load_config_json
from utils.determinism import set_deterministic
from utils.logging import get_json_logger, log_event
from utils.pdf import iter_pdf_to_images
from utils.text import detect_language_bucket
from validator import InvoiceFields, InvoiceOutput, PresenceBox
from architecture_diagram import generate_architecture_png


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
		to_stdout = bool(run_mode not in {"submission", "judge", "demo"})
		logger = get_json_logger("docai", log_path=output_dir / "pipeline_logs.jsonl", to_stdout=to_stdout)

		deadline = start_total + 30.0
		log_event(logger, "pipeline_start", doc_id=doc_id, pdf_path=str(pdf_path))

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
		pages = []
		skipped_pages = 0
		deadline_epoch = time.time() + max(0.0, 29.0 - (time.perf_counter() - start_total))
		for p in iter_pdf_to_images(pdf_path, dpi=config.dpi, max_pages=config.max_pages, deadline_epoch=deadline_epoch):
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
			dpi=config.dpi,
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
		allow_paddle_cfg = bool(getattr(config, "enable_paddleocr", False))
		allow_paddle = bool(allow_paddle_cfg or allow_paddle_env)
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
			"langs": tuple(getattr(config, "ocr_langs", ("en", "devanagari", "gujarati"))),
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
		t0 = time.perf_counter()
		dealer = extract_dealer_name(
			doc_layout,
			base_dir=base_dir,
			threshold=config.dealer_fuzzy_threshold,
			region_weight_overrides=getattr(config, "region_weight_overrides", None),
		)
		model = extract_model_name(
			doc_layout,
			base_dir=base_dir,
			region_weight_overrides=getattr(config, "region_weight_overrides", None),
		)
		hp = extract_horse_power(doc_layout, region_weight_overrides=getattr(config, "region_weight_overrides", None))
		asset = extract_asset_cost(doc_layout, region_weight_overrides=getattr(config, "region_weight_overrides", None))
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

		# 5) Vision detection (prefer last 2 pages)
		t0 = time.perf_counter()
		if time.perf_counter() >= deadline - 1.5:
			sig = None
			stamp = None
			vision_sec = 0.0
			log_event(logger, "latency_guard_skip_stage", stage="vision")
		else:
			detector = SignatureStampDetector(
				weights_path=config.yolo_weights_path,
				conf=config.yolo_conf,
				iou_thresh=getattr(config, "yolo_iou", 0.5),
				img_sizes=getattr(config, "yolo_img_sizes", (640,)),
			)
			detections = []
			imgs = page_images[-2:] if len(page_images) >= 2 else page_images[:1]
			for img in imgs:
				detections.extend(detector.detect(img))
			sig = best_by_label(detections, "signature")
			stamp = best_by_label(detections, "stamp")
			vision_sec = time.perf_counter() - t0
			log_event(
				logger,
				"stage_vision",
				sec=round(vision_sec, 4),
				yolo_used=bool(config.yolo_weights_path),
				signature_present=bool(sig),
				stamp_present=bool(stamp),
			)

		# 6) Assemble + validate
		t0 = time.perf_counter()
		fields = InvoiceFields(
			dealer_name=clean_str(dealer.value if isinstance(dealer.value, str) else None),
			model_name=clean_str(model.value if isinstance(model.value, str) else None),
			horse_power=clean_int(hp.value),
			asset_cost=clean_int(asset.value),
			signature=PresenceBox(present=bool(sig), bbox=sig.bbox.as_list() if sig else []),
			stamp=PresenceBox(present=bool(stamp), bbox=stamp.bbox.as_list() if stamp else []),
		)

		pages_n = len(page_images)
		yolo_used = config.yolo_weights_path is not None
		cost_breakdown = estimate_cost_breakdown(config, pages=pages_n, yolo_used=yolo_used)
		cost = float(cost_breakdown.get("total", estimate_cost(config, pages=pages_n, yolo_used=yolo_used)))

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
			"yolo_used": bool(config.yolo_weights_path),
			"signature_present": bool(sig),
			"stamp_present": bool(stamp),
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


def main() -> int:
	p = argparse.ArgumentParser(description="DocAI Invoice Extractor")
	p.add_argument("--pdf", required=True, help="Path to invoice PDF")
	p.add_argument("--doc-id", default=None, help="Doc id for output JSON")
	p.add_argument(
		"--config",
		default="",
		help="Optional JSON config (e.g. best_config.json). CLI flags override it.",
	)
	p.add_argument("--dpi", type=int, default=200)
	p.add_argument("--max-pages", type=int, default=5)
	p.add_argument(
		"--yolo-weights",
		default="",
		help="Optional YOLOv8 .pt weights for signature/stamp",
	)
	p.add_argument(
		"--enable-paddleocr",
		action="store_true",
		help="Enable PaddleOCR (slower, higher accuracy)",
	)
	p.add_argument("--out", default="outputs/result.json", help="Output JSON path (defaults under outputs/)")
	p.add_argument("--no-eda", action="store_true", help="Disable EDA artifact generation")
	p.add_argument("--no-error-report", action="store_true", help="Disable error report generation")
	p.add_argument("--no-diagram", action="store_true", help="Disable architecture diagram generation")
	args = p.parse_args()

	if bool(getattr(args, "enable_paddleocr", False)):
		os.environ["DOC_AI_ENABLE_PADDLEOCR"] = "1"

	base_dir = Path(__file__).resolve().parent
	doc_id = args.doc_id or Path(args.pdf).stem
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
		yolo_weights_path=Path(args.yolo_weights) if args.yolo_weights else None,
		enable_paddleocr=bool(getattr(args, "enable_paddleocr", False)),
	)
	try:
		result = run_pipeline(pdf_path=args.pdf, doc_id=doc_id, config=cfg)
	except BaseException as e:
		# Never hard-crash without writing output JSON (evaluator/demo friendliness)
		processing_time = float(round(0.0, 3))
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
		}
		try:
			InvoiceOutput.model_validate(fallback, strict=True)
		except Exception:
			pass
		print(f"[warn] run_pipeline failed for {doc_id}: {e}", file=sys.stderr)
		result = fallback

	output_dir = cfg.output_dir if Path(cfg.output_dir).is_absolute() else (base_dir / cfg.output_dir)

	# Resolve --out consistently under output_dir (independent of cwd)
	out_arg = Path(args.out)
	if out_arg.is_absolute():
		out_path = out_arg
	else:
		# If user passed outputs/... then drop leading outputs
		parts = list(out_arg.parts)
		if parts and parts[0] == "outputs":
			parts = parts[1:]
		out_path = output_dir.joinpath(*parts) if parts else (output_dir / "result.json")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

	# Judge-ready artifacts from run logs
	try:
		if not args.no_diagram:
			generate_architecture_png(output_dir / "architecture_diagram.png")
	except Exception as e:
		print(f"[warn] diagram generation failed: {e}", file=sys.stderr)
	try:
		if not args.no_eda:
			run_eda(runs_jsonl=output_dir / "runs.jsonl", outputs_dir=output_dir)
	except Exception as e:
		print(f"[warn] EDA generation failed: {e}", file=sys.stderr)
	try:
		if not args.no_error_report:
			run_error_analysis(runs_jsonl=output_dir / "runs.jsonl", outputs_dir=output_dir)
	except Exception as e:
		print(f"[warn] error analysis failed: {e}", file=sys.stderr)

	print(json.dumps(result, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
