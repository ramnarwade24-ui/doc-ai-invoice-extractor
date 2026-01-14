from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from ocr import OCRPage, OCRWord


def pymupdf_extract_words(pdf_path: str | Path, page_index: int) -> OCRPage:
	"""Fallback extractor for digital PDFs (not scanned images).

	Uses PyMuPDF's text extraction to return word boxes similar to OCR output.
	Confidence is heuristic (fixed high) since this is not OCR.
	"""
	pdf_path = Path(pdf_path)
	doc = fitz.open(str(pdf_path))
	page = doc.load_page(page_index)

	# words: x0, y0, x1, y1, "word", block_no, line_no, word_no
	words_raw = page.get_text("words") or []
	words: List[OCRWord] = []
	for (x0, y0, x1, y1, w, *_rest) in words_raw:
		text = str(w).strip()
		if not text:
			continue
		bbox = [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]
		words.append(OCRWord(text=text, bbox=bbox, conf=0.95))

	doc.close()
	return OCRPage(page_index=page_index, words=words)
