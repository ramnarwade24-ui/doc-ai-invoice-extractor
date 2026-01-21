#!/usr/bin/env python3
"""Offline EDA for jury evaluation (PNG invoices).

This script is **offline** and **optional**. It scans invoice PNGs under `data/images/`
(recursive, deterministically sorted) and writes:

- eda_report.json (rich metrics + heatmap)
- eda_summary.csv (small tabular summary)
- eda_profile.json (lightweight profile consumed by extraction)

Nothing in this module is imported by the extraction pipeline.

Usage:
  python eda.py --images data/images --out outputs/eda
  python eda.py --plot outputs/eda/eda_report.json --pdf outputs/eda/eda_report.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


DEFAULT_KEYWORDS = [
	"dealer",
	"seller",
	"vendor",
	"supplier",
	"invoice",
	"invoice no",
	"invoice number",
	"date",
	"total",
	"grand total",
	"net amount",
	"amount",
	"asset cost",
	"model",
	"hp",
	"horse power",
	"signature",
	"stamp",
]


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _discover_pngs(images: str | Path, *, repo_root: Path) -> List[Path]:
	img_str = str(images)
	img_path = Path(img_str)
	resolved = img_path if img_path.is_absolute() else (repo_root / img_path)

	if resolved.exists() and resolved.is_file():
		return [resolved] if resolved.suffix.lower() == ".png" else []
	if resolved.exists() and resolved.is_dir():
		return sorted([p for p in resolved.rglob("*.png") if p.is_file()])

	return sorted([p for p in repo_root.glob(img_str) if p.is_file() and p.suffix.lower() == ".png"])


def _maybe_import_docai() -> None:
	"""Allow importing doc_ai_project modules when running from repo root."""
	repo_root = _repo_root()
	doc_ai_dir = repo_root / "doc_ai_project"
	if str(doc_ai_dir) not in sys.path:
		sys.path.insert(0, str(doc_ai_dir))


def _safe_median(xs: List[float]) -> float:
	if not xs:
		return 0.0
	try:
		return float(statistics.median(xs))
	except Exception:
		xs2 = sorted(float(x) for x in xs)
		return float(xs2[len(xs2) // 2])


def _safe_mean(xs: List[float]) -> float:
	if not xs:
		return 0.0
	return float(sum(xs) / max(1, len(xs)))


def _normalize_text_for_keywords(text: str) -> str:
	return (text or "").casefold()


def _count_keywords(text: str, keywords: List[str]) -> Dict[str, int]:
	t = _normalize_text_for_keywords(text)
	out: Dict[str, int] = {}
	for kw in keywords:
		k = (kw or "").strip().casefold()
		if not k:
			continue
		out[kw] = int(t.count(k))
	return out


def _heatmap_grid(grid: int) -> List[List[int]]:
	return [[0 for _ in range(grid)] for __ in range(grid)]


def _clamp(v: int, lo: int, hi: int) -> int:
	return lo if v < lo else hi if v > hi else v


def _accumulate_heatmap(
	*,
	heat: List[List[int]],
	grid: int,
	word_bboxes_xyxy: Iterable[Tuple[int, int, int, int]],
	img_w: int,
	img_h: int,
) -> None:
	if img_w <= 0 or img_h <= 0:
		return
	for (x1, y1, x2, y2) in word_bboxes_xyxy:
		cx = int((x1 + x2) * 0.5)
		cy = int((y1 + y2) * 0.5)
		gx = _clamp(int((cx / img_w) * grid), 0, grid - 1)
		gy = _clamp(int((cy / img_h) * grid), 0, grid - 1)
		heat[gy][gx] += 1


def _bbox_xyxy_from_paddle_box(box: object) -> Optional[Tuple[int, int, int, int]]:
	"""PaddleOCR word bbox is typically 4 points; we convert to xyxy."""
	try:
		pts = list(box)  # type: ignore[arg-type]
		if len(pts) != 4:
			return None
		xs = [float(p[0]) for p in pts]
		ys = [float(p[1]) for p in pts]
		x1, x2 = int(min(xs)), int(max(xs))
		y1, y2 = int(min(ys)), int(max(ys))
		return x1, y1, x2, y2
	except Exception:
		return None


@dataclass(frozen=True)
class ImageEDA:
	path: str
	width: int
	height: int
	avg_ocr_conf: float
	word_count: int
	language: str
	top_heavy_ratio: float
	keyword_counts: Dict[str, int]


def run_eda(
	*,
	images_dir: Path,
	out_dir: Path,
	grid_size: int = 64,
	keywords: Optional[List[str]] = None,
	max_images: int = 0,
	no_ocr: bool = False,
) -> Dict[str, object]:
	"""Compute EDA report dict and write artifacts."""
	keywords = keywords or list(DEFAULT_KEYWORDS)
	images = _discover_pngs(images_dir, repo_root=_repo_root())
	if max_images and max_images > 0:
		images = images[: int(max_images)]

	out_dir.mkdir(parents=True, exist_ok=True)

	heat = _heatmap_grid(int(grid_size))
	rows: List[ImageEDA] = []
	lang_counts: Dict[str, int] = {}
	kw_counts_total: Dict[str, int] = {k: 0 for k in keywords}

	# Optional OCR engine (offline-only)
	ocr_engine = None
	detect_language_bucket = None
	normalize_keywords = None
	if not no_ocr:
		try:
			_maybe_import_docai()
			from ocr import PaddleOCREngine  # type: ignore
			from utils.text import detect_language_bucket as _dlb, normalize_keywords as _nk  # type: ignore

			ocr_engine = PaddleOCREngine(
				use_angle_cls=True,
				langs=("en", "devanagari", "gujarati"),
				max_retries=1,
				autorotate=True,
				adaptive_threshold=False,
				shadow_remove=True,
				perspective_correct=False,
				upscale_if_low_res=True,
			)
			detect_language_bucket = _dlb
			normalize_keywords = _nk
		except Exception:
			ocr_engine = None

	for img_path in images:
		try:
			img = Image.open(img_path).convert("RGB")
		except Exception:
			continue
		w, h = int(img.width), int(img.height)

		avg_conf = 0.0
		word_count = 0
		language = "unknown"
		top_heavy_ratio = 0.0
		kw_counts: Dict[str, int] = {k: 0 for k in keywords}

		if ocr_engine is not None and detect_language_bucket is not None:
			try:
				page = ocr_engine.run_page(0, img)
				texts = [getattr(wd, "text", "") for wd in getattr(page, "words", [])]
				confs = [float(getattr(wd, "conf", 0.0) or 0.0) for wd in getattr(page, "words", [])]
				word_count = int(len(texts))
				avg_conf = float(sum(confs) / max(1, len(confs))) if confs else 0.0

				# Language bucket (uses normalized keywords + unicode hints)
				try:
					norm_texts = [normalize_keywords(t) for t in texts] if normalize_keywords else texts
				except Exception:
					norm_texts = texts
				language = str(detect_language_bucket(norm_texts) or "unknown")

				# Keyword frequency from whole-page text
				whole = " ".join(norm_texts)
				kw_counts = _count_keywords(whole, keywords)

				# Heatmap + top-heavy ratio
				boxes_xyxy = []
				top_cut = 0.25 * float(h)
				top_hits = 0
				for wd in getattr(page, "words", []):
					bb = _bbox_xyxy_from_paddle_box(getattr(wd, "bbox", None))
					if not bb:
						continue
					(x1, y1, x2, y2) = bb
					boxes_xyxy.append(bb)
					cy = 0.5 * float(y1 + y2)
					if cy <= top_cut:
						top_hits += 1
				if word_count > 0:
					top_heavy_ratio = float(top_hits / max(1, word_count))
				_accumulate_heatmap(heat=heat, grid=int(grid_size), word_bboxes_xyxy=boxes_xyxy, img_w=w, img_h=h)
			except Exception:
				# Keep EDA resilient; record what we can.
				pass

		lang_counts[language] = int(lang_counts.get(language, 0) + 1)
		for k, v in kw_counts.items():
			kw_counts_total[k] = int(kw_counts_total.get(k, 0) + int(v))

		rows.append(
			ImageEDA(
				path=str(img_path),
				width=w,
				height=h,
				avg_ocr_conf=float(avg_conf),
				word_count=int(word_count),
				language=language,
				top_heavy_ratio=float(top_heavy_ratio),
				keyword_counts=dict(kw_counts),
			)
		)

	widths = [float(r.width) for r in rows]
	heights = [float(r.height) for r in rows]
	areas = [float(r.width * r.height) for r in rows]
	confs = [float(r.avg_ocr_conf) for r in rows if r.word_count > 0]
	top_ratios = [float(r.top_heavy_ratio) for r in rows if r.word_count > 0]

	# Heuristics for profile
	low_res_rate = 0.0
	if rows:
		low_res = 0
		for r in rows:
			if min(r.width, r.height) < 900 or (r.width * r.height) < 1_000_000:
				low_res += 1
		low_res_rate = float(low_res / max(1, len(rows)))

	hindi_rate = 0.0
	if rows:
		hi = 0
		for r in rows:
			if str(r.language).lower() in {"hi", "hindi", "devanagari"}:
				hi += 1
		hindi_rate = float(hi / max(1, len(rows)))

	noisy_rate = 0.0
	if rows:
		noisy = 0
		for r in rows:
			# Noisy: low OCR confidence OR very low word count (typical for poor scans)
			if (r.word_count > 0 and r.avg_ocr_conf < 0.60) or (r.word_count > 0 and r.word_count < 10):
				noisy += 1
		noisy_rate = float(noisy / max(1, len(rows)))

	top_heavy_rate = 0.0
	if rows:
		top = 0
		for r in rows:
			if r.word_count > 0 and r.top_heavy_ratio >= 0.55:
				top += 1
		top_heavy_rate = float(top / max(1, len(rows)))

	rec_upscale = 1.0
	if low_res_rate >= 0.20:
		rec_upscale = 1.5
	if low_res_rate >= 0.45:
		rec_upscale = 2.0

	profile = {
		"version": 1,
		"dataset": {
			"image_count": int(len(rows)),
			"low_res_rate": float(round(low_res_rate, 4)),
			"hindi_rate": float(round(hindi_rate, 4)),
			"noisy_rate": float(round(noisy_rate, 4)),
			"top_heavy_rate": float(round(top_heavy_rate, 4)),
		},
		"flags": {
			"top_heavy_text": bool(top_heavy_rate >= 0.50),
			"noisy_layout": bool(noisy_rate >= 0.30),
			"hindi_detected": bool(hindi_rate >= 0.05),
		},
		"recommendations": {
			"upscale_factor": float(rec_upscale),
			"ocr_langs": (
				["devanagari", "en", "gujarati"]
				if bool(hindi_rate >= 0.05)
				else ["en", "devanagari", "gujarati"]
			),
			"header_first": bool(top_heavy_rate >= 0.50),
			"keyword_anchored": bool(noisy_rate >= 0.30),
		},
	}

	report: Dict[str, object] = {
		"version": 1,
		"input": {"images": str(images_dir), "png_count": int(len(rows))},
		"image_size": {
			"width_min": int(min(widths) if widths else 0),
			"width_median": int(_safe_median(widths)),
			"width_max": int(max(widths) if widths else 0),
			"height_min": int(min(heights) if heights else 0),
			"height_median": int(_safe_median(heights)),
			"height_max": int(max(heights) if heights else 0),
			"area_median": int(_safe_median(areas)),
		},
		"language_distribution": dict(sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
		"keyword_frequency": dict(sorted(kw_counts_total.items(), key=lambda kv: (-kv[1], kv[0]))),
		"layout_heatmap": {"grid_size": int(grid_size), "counts": heat},
		"quality": {
			"avg_ocr_conf_median": float(round(_safe_median(confs), 4)),
			"avg_ocr_conf_mean": float(round(_safe_mean(confs), 4)),
			"top_heavy_ratio_median": float(round(_safe_median(top_ratios), 4)),
		},
		"profile": profile,
		"samples": [
			{
				"path": r.path,
				"width": int(r.width),
				"height": int(r.height),
				"avg_ocr_conf": float(round(r.avg_ocr_conf, 4)),
				"word_count": int(r.word_count),
				"language": r.language,
				"top_heavy_ratio": float(round(r.top_heavy_ratio, 4)),
			}
			for r in rows[: min(200, len(rows))]
		],
	}

	(out_dir / "eda_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	(out_dir / "eda_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

	# CSV summary (one row)
	csv_path = out_dir / "eda_summary.csv"
	with csv_path.open("w", encoding="utf-8", newline="") as f:
		wtr = csv.writer(f)
		wtr.writerow(
			[
				"image_count",
				"width_median",
				"height_median",
				"area_median",
				"avg_ocr_conf_median",
				"top_heavy_ratio_median",
				"low_res_rate",
				"hindi_rate",
				"noisy_rate",
				"top_heavy_rate",
				"recommended_upscale_factor",
			]
		)
		wtr.writerow(
			[
				int(len(rows)),
				int(_safe_median(widths)),
				int(_safe_median(heights)),
				int(_safe_median(areas)),
				float(round(_safe_median(confs), 6)),
				float(round(_safe_median(top_ratios), 6)),
				float(round(low_res_rate, 6)),
				float(round(hindi_rate, 6)),
				float(round(noisy_rate, 6)),
				float(round(top_heavy_rate, 6)),
				float(round(rec_upscale, 3)),
			]
		)

	return report


def plot_report_to_pdf(*, report_path: Path, pdf_path: Path) -> None:
	"""Generate eda_report.pdf from eda_report.json (no notebook required)."""
	try:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		from matplotlib.backends.backend_pdf import PdfPages
	except Exception as e:
		raise RuntimeError(f"Plotting requires matplotlib: {e}")

	obj = json.loads(report_path.read_text(encoding="utf-8"))
	img = obj.get("image_size") or {}
	langs = obj.get("language_distribution") or {}
	kws = obj.get("keyword_frequency") or {}
	heat = (obj.get("layout_heatmap") or {}).get("counts")
	grid = int((obj.get("layout_heatmap") or {}).get("grid_size") or 0)

	# For histogram, use samples' width/height (deterministic subset) when available.
	samples = obj.get("samples") or []
	widths = [int(s.get("width") or 0) for s in samples if isinstance(s, dict)]
	heights = [int(s.get("height") or 0) for s in samples if isinstance(s, dict)]

	pdf_path.parent.mkdir(parents=True, exist_ok=True)
	with PdfPages(str(pdf_path)) as pdf:
		# Image size histogram
		fig = plt.figure(figsize=(8.5, 5))
		plt.title("Image size distribution (width)")
		plt.hist([w for w in widths if w > 0], bins=20)
		plt.xlabel("Width (px)")
		plt.ylabel("Count")
		plt.grid(True, alpha=0.2)
		pdf.savefig(fig)
		plt.close(fig)

		fig = plt.figure(figsize=(8.5, 5))
		plt.title("Image size distribution (height)")
		plt.hist([h for h in heights if h > 0], bins=20)
		plt.xlabel("Height (px)")
		plt.ylabel("Count")
		plt.grid(True, alpha=0.2)
		pdf.savefig(fig)
		plt.close(fig)

		# Keyword frequency bar chart (top 15)
		items = [(str(k), int(v)) for k, v in (kws.items() if isinstance(kws, dict) else [])]
		items.sort(key=lambda kv: kv[1], reverse=True)
		items = items[:15]
		fig = plt.figure(figsize=(10, 6))
		plt.title("Keyword frequency (top 15)")
		plt.bar([k for k, _ in items], [v for _, v in items])
		plt.xticks(rotation=45, ha="right")
		plt.ylabel("Count")
		plt.tight_layout()
		pdf.savefig(fig)
		plt.close(fig)

		# Language pie chart
		lang_items = [(str(k), int(v)) for k, v in (langs.items() if isinstance(langs, dict) else [])]
		lang_items.sort(key=lambda kv: kv[1], reverse=True)
		fig = plt.figure(figsize=(8, 8))
		plt.title("Language distribution")
		if lang_items:
			plt.pie([v for _, v in lang_items], labels=[k for k, _ in lang_items], autopct="%1.1f%%")
		else:
			plt.text(0.5, 0.5, "No OCR language data", ha="center", va="center")
		pdf.savefig(fig)
		plt.close(fig)

		# Heatmap
		fig = plt.figure(figsize=(8, 8))
		plt.title("Text location heatmap")
		if isinstance(heat, list) and grid > 0:
			import numpy as np

			arr = np.array(heat, dtype=float)
			plt.imshow(arr, cmap="hot", origin="upper")
			plt.colorbar(fraction=0.046, pad=0.04)
			plt.axis("off")
		else:
			plt.text(0.5, 0.5, "No heatmap data", ha="center", va="center")
		pdf.savefig(fig)
		plt.close(fig)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Offline EDA for PNG invoices")
	p.add_argument("--images", default="data/images", help="Folder/glob for invoice PNGs (recursive if directory)")
	p.add_argument("--out", default="outputs/eda", help="Output directory for EDA artifacts")
	p.add_argument("--grid", type=int, default=64, help="Heatmap grid size (NxN)")
	p.add_argument("--max-images", type=int, default=0, help="Optional cap for faster local EDA (0 = all)")
	p.add_argument("--no-ocr", action="store_true", help="Skip OCR-derived metrics (language, keywords, heatmap)")
	p.add_argument("--plot", default="", help="Path to eda_report.json to plot")
	p.add_argument("--pdf", default="", help="Path to output eda_report.pdf")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = _repo_root()

	if args.plot:
		report_path = Path(args.plot)
		if not report_path.is_absolute():
			report_path = repo_root / report_path
		pdf_path = Path(args.pdf or "eda_report.pdf")
		if not pdf_path.is_absolute():
			pdf_path = report_path.parent / pdf_path
		plot_report_to_pdf(report_path=report_path, pdf_path=pdf_path)
		print(json.dumps({"ok": True, "pdf": str(pdf_path)}, indent=2))
		return 0

	images_dir = Path(args.images)
	if not images_dir.is_absolute():
		images_dir = repo_root / images_dir
	out_dir = Path(args.out)
	if not out_dir.is_absolute():
		out_dir = repo_root / out_dir

	report = run_eda(
		images_dir=images_dir,
		out_dir=out_dir,
		grid_size=int(args.grid),
		max_images=int(args.max_images),
		no_ocr=bool(args.no_ocr),
	)
	print(json.dumps({"ok": True, "out": str(out_dir), "png_count": int((report.get("input") or {}).get("png_count") or 0)}, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
