from __future__ import annotations

import csv
import io
import json
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from utils.config import PipelineConfig
from utils.config_io import load_config_json
from utils.text import normalize_name
from validator import InvoiceOutput


def warmup_ocr(
	*,
	seed: int = 1337,
	config_path: Optional[str | Path] = None,
	mode: Literal["fast", "accurate"] = "fast",
) -> bool:
	"""Warm up PaddleOCR once (in an isolated process).

	This prevents first-run model downloads from happening inside the timed demo loop.
	Never raises; returns True/False.
	"""
	# Do not warm up automatically in demo/judge flows. Warmup can trigger heavy imports/downloads
	# and is not required for a crash-safe pipeline (we fall back if PaddleOCR fails).
	return False
	try:
		from pipeline.ocr_engine import warmup_ocr as _warmup

		cfg = _load_config(config_path, seed=int(seed), mode=mode)
		engine_kwargs = {
			"use_angle_cls": bool(getattr(cfg, "use_angle_cls", True)),
			"langs": tuple(getattr(cfg, "ocr_langs", ("en", "devanagari", "gujarati"))),
			"max_retries": int(getattr(cfg, "max_ocr_retries", 1)),
			"preprocess_variants": [tuple(v) for v in getattr(cfg, "ocr_preprocess_variants", ())] or None,
			"autorotate": bool(getattr(cfg, "ocr_autorotate", False)),
			"adaptive_threshold": bool(getattr(cfg, "ocr_adaptive_threshold", False)),
			"shadow_remove": bool(getattr(cfg, "ocr_shadow_remove", True)),
			"perspective_correct": bool(getattr(cfg, "ocr_perspective_correct", False)),
			"upscale_if_low_res": bool(getattr(cfg, "ocr_upscale_if_low_res", True)),
		}
		return bool(_warmup(seed=int(seed), engine_kwargs=engine_kwargs))
	except Exception:
		return False


def _repo_root() -> Path:
	# doc_ai_project/utils/demo_utils.py -> repo root is two levels up
	return Path(__file__).resolve().parents[2]


def discover_pdfs(path: str | Path, recursive: bool = True, limit: Optional[int] = None, seed: int = 1337) -> List[Path]:
	"""Discover PDFs deterministically.

	- If `path` is a PDF file: returns [path]
	- If `path` is a directory: finds PDFs (recursive by default)
	- Otherwise: treated as a glob pattern relative to repo root

	If limit is set, returns a deterministic sample of size `limit`.
	"""
	root = _repo_root()
	p = Path(str(path))
	resolved = p if p.is_absolute() else (root / p)

	pdfs: List[Path] = []
	if resolved.exists() and resolved.is_file():
		pdfs = [resolved] if resolved.suffix.lower() == ".pdf" else []
	elif resolved.exists() and resolved.is_dir():
		if recursive:
			pdfs = [q for q in resolved.rglob("*.pdf") if q.is_file()]
		else:
			pdfs = [q for q in resolved.glob("*.pdf") if q.is_file()]
	else:
		# glob pattern relative to repo root
		pdfs = [q for q in root.glob(str(path)) if q.is_file() and q.suffix.lower() == ".pdf"]

	pdfs = sorted(pdfs)
	if limit is None:
		return pdfs

	k = max(0, int(limit))
	if k <= 0:
		return []
	if k >= len(pdfs):
		return pdfs

	rng = random.Random(int(seed))
	sample = rng.sample(pdfs, k=k)
	return sample


def _resolve_config_path(config_path: Optional[str | Path]) -> Optional[Path]:
	base_dir = Path(__file__).resolve().parents[1]  # doc_ai_project/
	if not config_path:
		p = base_dir / "best_config.json"
		return p if p.exists() else None
	p = Path(str(config_path))
	if p.is_absolute():
		return p
	# Accept either "best_config.json" (doc_ai_project relative) or repo-root-relative
	cand1 = base_dir / p
	if cand1.exists():
		return cand1
	cand2 = _repo_root() / p
	if cand2.exists():
		return cand2
	return cand1



def _load_config(config_path: Optional[str | Path], *, seed: int, mode: Literal["fast", "accurate"]) -> PipelineConfig:
	cfgp = _resolve_config_path(config_path)
	cfg = load_config_json(cfgp) if (cfgp and cfgp.exists()) else PipelineConfig()
	# Always deterministic for demo/judge mode.
	allow_paddle_env = str(os.getenv("DOC_AI_ENABLE_PADDLEOCR", "0")).strip().lower() in {"1", "true", "yes"}
	base = replace(
		cfg,
		deterministic=True,
		run_mode="submission",
		seed=int(seed),
		enable_paddleocr=bool(allow_paddle_env),
	)
	if mode == "accurate":
		return base
	# mode == "fast" (default)
	# Goal: keep `demo_runner --limit 10` under 30s in constrained environments.
	return replace(
		base,
		dpi=min(int(getattr(base, "dpi", 200)), 150),
		max_pages=min(int(getattr(base, "max_pages", 5)), 1),
		save_overlays=False,
		yolo_weights_path=None,
	)


def run_pipeline(
	pdf_paths: Sequence[Path],
	config_path: Optional[str | Path] = None,
	*,
	seed: int = 1337,
	mode: Literal["fast", "accurate"] = "fast",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	"""Run the in-process pipeline on multiple PDFs.

	Returns (results, errors).
	"""
	from executable import run_pipeline as _run

	cfg = _load_config(config_path, seed=int(seed), mode=mode)
	results: List[Dict[str, Any]] = []
	errors: List[Dict[str, Any]] = []

	for pdf in pdf_paths:
		doc_id = pdf.stem
		try:
			out_obj = _run(pdf_path=str(pdf), doc_id=doc_id, config=cfg)
			InvoiceOutput.model_validate(out_obj, strict=True)
			fields = out_obj.get("fields") or {}
			results.append(
				{
					"doc_id": doc_id,
					"pdf": str(pdf),
					"dealer_name": fields.get("dealer_name"),
					"model_name": fields.get("model_name"),
					"horse_power": fields.get("horse_power"),
					"asset_cost": fields.get("asset_cost"),
					"signature_present": bool((fields.get("signature") or {}).get("present")) if isinstance(fields.get("signature"), dict) else bool(fields.get("signature")),
					"stamp_present": bool((fields.get("stamp") or {}).get("present")) if isinstance(fields.get("stamp"), dict) else bool(fields.get("stamp")),
					"confidence": out_obj.get("confidence"),
					"processing_time_sec": out_obj.get("processing_time_sec"),
					"cost_estimate_usd": out_obj.get("cost_estimate_usd"),
					"review_required": out_obj.get("review_required"),
				}
			)
		except Exception as e:
			errors.append({"doc_id": doc_id, "pdf": str(pdf), "error": str(e)})

	return results, errors


def write_csv(results: Sequence[Dict[str, Any]], out_path: str | Path) -> None:
	path = Path(str(out_path))
	if not path.is_absolute():
		path = _repo_root() / path
	path.parent.mkdir(parents=True, exist_ok=True)

	if not results:
		path.write_text("", encoding="utf-8")
		return

	with path.open("w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
		w.writeheader()
		for r in results:
			w.writerow(r)


def write_json(report: Dict[str, Any], out_path: str | Path) -> None:
	path = Path(str(out_path))
	if not path.is_absolute():
		path = _repo_root() / path
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def format_table(results: Sequence[Dict[str, Any]]) -> str:
	cols = [
		("dealer_name", "dealer_name"),
		("model_name", "model_name"),
		("horse_power", "hp"),
		("asset_cost", "asset_cost"),
		("signature_present", "signature"),
		("stamp_present", "stamp"),
		("confidence", "confidence"),
		("processing_time_sec", "latency"),
	]

	def fmt(v: Any) -> str:
		if v is None:
			return ""
		if isinstance(v, float):
			return f"{v:.3f}"
		return str(v)

	widths = {label: len(label) for _, label in cols}
	for r in results:
		for key, label in cols:
			widths[label] = min(60, max(widths[label], len(fmt(r.get(key)))))

	header = " | ".join(label.ljust(widths[label]) for _, label in cols)
	sep = "-+-".join("-" * widths[label] for _, label in cols)
	lines = [header, sep]
	for r in results:
		lines.append(" | ".join(fmt(r.get(key)).ljust(widths[label])[: widths[label]] for key, label in cols))
	return "\n".join(lines)


def load_label_fields(labels_dir: Path, doc_id: str) -> Optional[Dict[str, Any]]:
	p = labels_dir / f"{doc_id}.json"
	if not p.exists():
		return None
	try:
		obj = json.loads(p.read_text(encoding="utf-8"))
		if isinstance(obj, dict) and "fields" in obj and isinstance(obj["fields"], dict):
			return obj["fields"]
		if isinstance(obj, dict):
			return obj
		return None
	except Exception:
		return None


def normalized_equal(a: Any, b: Any) -> bool:
	return normalize_name(str(a or "")) == normalize_name(str(b or ""))
