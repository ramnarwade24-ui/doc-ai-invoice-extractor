#!/usr/bin/env python3
"""Extract full OCR text from a PDF or image.

This is a lightweight utility to dump the raw OCR text that the project uses.

Examples:
  # PDF (digital PDFs work out-of-the-box via PyMuPDF text extraction)
  python extract_text.py --pdf data/pdfs/invoice_001.pdf --out outputs/ocr_text.txt

  # PNG/JPG (requires PaddleOCR installed; may take longer)
  python extract_text.py --png data/images/172427893_3_pg11.png --out outputs/ocr_text.txt

	# PNG/JPG (prefer Tesseract if installed)
	python extract_text.py --png data/images/172427893_3_pg11.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _ensure_docai_on_path(repo_root: Path) -> Path:
	doc_ai_dir = repo_root / "doc_ai_project"
	sys.path.insert(0, str(doc_ai_dir))
	return doc_ai_dir


def _page_text_from_ocr_page(page, page_size: tuple[int, int]) -> str:
	from layout import build_structured_layout

	sl = build_structured_layout(page, page_size=page_size)
	# Use line grouping order for a stable, human-readable dump.
	return "\n".join([ln.text for ln in sl.all_lines if (ln.text or "").strip()])


def _extract_text_from_pdf(pdf_path: Path, *, max_pages: int | None) -> str:
	import fitz  # PyMuPDF

	from ocr_fallback import pymupdf_extract_words
	from pipeline.ocr_engine import OCRFailure, ocr_page_with_timeout, warmup_ocr
	from utils.pdf import iter_pdf_to_images

	# Decide whether it is a digital PDF by checking for any selectable text.
	is_digital_pdf = False
	with fitz.open(str(pdf_path)) as doc:
		n_pages = int(doc.page_count)
		probe_pages = min(2, n_pages)
		for i in range(probe_pages):
			try:
				page = doc.load_page(i)
				if (page.get_text("text") or "").strip():
					is_digital_pdf = True
					break
			except Exception:
				pass

	# If digital, just use word boxes from PyMuPDF (fast and deterministic).
	if is_digital_pdf:
		chunks: list[str] = []
		with fitz.open(str(pdf_path)) as doc:
			n_pages = int(doc.page_count)
			limit = min(n_pages, int(max_pages)) if max_pages is not None else n_pages
			for i in range(limit):
				p = doc.load_page(i)
				w = int(p.rect.width)
				h = int(p.rect.height)
				ocr_page = pymupdf_extract_words(str(pdf_path), i)
				chunks.append(_page_text_from_ocr_page(ocr_page, (w, h)))
		return "\n\n".join([c for c in chunks if c.strip()])

	# Otherwise, render pages and run PaddleOCR.
	# Note: PaddleOCR is disabled by default in the main pipeline; this utility
	# calls OCR directly. If PaddleOCR isn't available, we surface a clear error.
	warmed = warmup_ocr(seed=1337, engine_kwargs=None, timeout_sec=20.0)
	if not warmed:
		# continue anyway; ocr_page_with_timeout may still work if worker can start
		pass

	chunks: list[str] = []
	deadline_epoch = None
	count = 0
	for p in iter_pdf_to_images(str(pdf_path), dpi=250, max_pages=max_pages or 50, deadline_epoch=deadline_epoch):
		count += 1
		try:
			ocr_page = ocr_page_with_timeout(page_index=int(p.page_index), image=p.image, seed=1337, engine_kwargs=None, timeout_sec=15.0)
		except OCRFailure as e:
			raise RuntimeError(
				"PaddleOCR failed or is unavailable. "
				"If you are running the full pipeline, set DOC_AI_ENABLE_PADDLEOCR=1. "
				f"Error: {e}"
			)
		page_size = (int(p.image.size[0]), int(p.image.size[1]))
		chunks.append(_page_text_from_ocr_page(ocr_page, page_size))
		if max_pages is not None and count >= int(max_pages):
			break

	return "\n\n".join([c for c in chunks if c.strip()])


def _extract_text_from_image(image_path: Path) -> str:
	from PIL import Image

	# Prefer Tesseract for stability in constrained environments.
	from ocr import tesseract_run_page_best
	from pipeline.ocr_engine import OCRFailure, ocr_page_with_timeout, warmup_ocr

	img = Image.open(str(image_path)).convert("RGB")
	# Tesseract path
	try:
		ocr_page = tesseract_run_page_best(0, img, langs=["en"])
		if getattr(ocr_page, "words", None):
			page_size = (int(img.size[0]), int(img.size[1]))
			return _page_text_from_ocr_page(ocr_page, page_size)
	except Exception:
		pass

	# PaddleOCR path (fallback)
	warmed = warmup_ocr(seed=1337, engine_kwargs=None, timeout_sec=20.0)
	if not warmed:
		# continue anyway
		pass
	try:
		ocr_page = ocr_page_with_timeout(page_index=0, image=img, seed=1337, engine_kwargs=None, timeout_sec=20.0)
	except OCRFailure as e:
		raise RuntimeError(
			"PaddleOCR failed or is unavailable for image OCR. "
			"If you are running the full pipeline, set DOC_AI_ENABLE_PADDLEOCR=1. "
			f"Error: {e}"
		)
	page_size = (int(img.size[0]), int(img.size[1]))
	return _page_text_from_ocr_page(ocr_page, page_size)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Dump full OCR text from a PDF or image")
	g = p.add_mutually_exclusive_group(required=True)
	g.add_argument("--pdf", type=str, help="Path to a PDF")
	g.add_argument("--png", type=str, help="Path to a PNG/JPG image")
	p.add_argument("--max-pages", type=int, default=None, help="Max pages for PDFs")
	p.add_argument("--out", type=str, default=None, help="Output text file (defaults to stdout)")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = _repo_root()
	_ensure_docai_on_path(repo_root)

	try:
		if args.pdf:
			text = _extract_text_from_pdf(Path(args.pdf), max_pages=args.max_pages)
		else:
			text = _extract_text_from_image(Path(args.png))
	except Exception as e:
		print(f"[FAIL] {e}", file=sys.stderr)
		return 2

	if args.out:
		out_path = Path(args.out)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_path.write_text(text, encoding="utf-8")
		print(str(out_path))
		return 0

	print(text)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
