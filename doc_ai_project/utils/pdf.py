from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

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
	mat = fitz.Matrix(zoom, zoom)

	page_count = doc.page_count
	limit = page_count if max_pages is None else min(page_count, max_pages)

	for i in range(limit):
		page = doc.load_page(i)
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
	mat = fitz.Matrix(zoom, zoom)

	page_count = doc.page_count
	limit = page_count if max_pages is None else min(page_count, max_pages)

	for i in range(limit):
		if deadline_epoch is not None and time.time() >= deadline_epoch:
			break
		page = doc.load_page(i)
		pix = page.get_pixmap(matrix=mat, alpha=False)
		img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
		yield PageImage(page_index=i, image=img)

	doc.close()
