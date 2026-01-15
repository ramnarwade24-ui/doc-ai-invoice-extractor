from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import math
import os
import time

import fitz  # PyMuPDF
from PIL import Image


@dataclass(frozen=True)
class PageImage:
	page_index: int
	image: Image.Image


def pdf_to_images(pdf_path: str | Path, dpi: int = 200, max_pages: int | None = None) -> List[PageImage]:
	pdf_path = Path(pdf_path)
	doc = fitz.open(str(pdf_path))
	images: List[PageImage] = []

	zoom = dpi / 72.0
	max_pixels = int(os.getenv("DOC_AI_MAX_IMAGE_PIXELS", "20000000"))  # ~20MP
	max_side = int(os.getenv("DOC_AI_MAX_IMAGE_SIDE", "8000"))

	page_count = doc.page_count
	limit = page_count if max_pages is None else min(page_count, max_pages)

	for i in range(limit):
		page = doc.load_page(i)
		# Memory safety: cap rasterization size for very large pages.
		# Some PDFs can have huge page dimensions; rendering at high DPI can OOM and get the process SIGTERM'd.
		rect = page.rect
		w = float(rect.width) * float(zoom)
		h = float(rect.height) * float(zoom)
		scale = 1.0
		try:
			if w > 0 and h > 0:
				if (w * h) > max_pixels or max(w, h) > max_side:
					scale_pix = math.sqrt(max(1.0, float(max_pixels)) / max(1.0, (w * h)))
					scale_side = float(max_side) / max(1.0, max(w, h))
					scale = max(0.05, min(1.0, scale_pix, scale_side))
		except Exception:
			scale = 1.0
		mat = fitz.Matrix(float(zoom) * float(scale), float(zoom) * float(scale))
		pix = page.get_pixmap(matrix=mat, alpha=False)
		mode = "RGB"
		img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
		images.append(PageImage(page_index=i, image=img))

	return images


def iter_pdf_to_images(
	pdf_path: str | Path,
	dpi: int = 200,
	max_pages: Optional[int] = None,
	deadline_epoch: Optional[float] = None,
) -> Iterator[PageImage]:
	"""Yield page images one-by-one, optionally stopping when time deadline is reached."""
	pdf_path = Path(pdf_path)
	doc = fitz.open(str(pdf_path))

	zoom = dpi / 72.0
	max_pixels = int(os.getenv("DOC_AI_MAX_IMAGE_PIXELS", "20000000"))  # ~20MP
	max_side = int(os.getenv("DOC_AI_MAX_IMAGE_SIDE", "8000"))

	page_count = doc.page_count
	limit = page_count if max_pages is None else min(page_count, max_pages)

	for i in range(limit):
		if deadline_epoch is not None and time.time() >= deadline_epoch:
			break
		page = doc.load_page(i)
		# Memory safety: adapt DPI downwards for very large pages.
		rect = page.rect
		w = float(rect.width) * float(zoom)
		h = float(rect.height) * float(zoom)
		scale = 1.0
		try:
			if w > 0 and h > 0:
				if (w * h) > max_pixels or max(w, h) > max_side:
					scale_pix = math.sqrt(max(1.0, float(max_pixels)) / max(1.0, (w * h)))
					scale_side = float(max_side) / max(1.0, max(w, h))
					scale = max(0.05, min(1.0, scale_pix, scale_side))
		except Exception:
			scale = 1.0
		mat = fitz.Matrix(float(zoom) * float(scale), float(zoom) * float(scale))
		pix = page.get_pixmap(matrix=mat, alpha=False)
		img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
		yield PageImage(page_index=i, image=img)

	doc.close()
