from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def generate_sample_invoice_pdf(out_path: Path) -> Path:
	"""Generate a digital-text sample invoice PDF for smoke-testing.

	This is intentionally a text PDF (not an embedded image) so the pipeline can
	run end-to-end even when falling back to PyMuPDF text extraction.
	"""
	out_path.parent.mkdir(parents=True, exist_ok=True)

	doc = fitz.open()
	page = doc.new_page(width=595, height=842)  # A4 points

	x, y = 50, 60
	line_h = 18
	font_size = 12

	lines = [
		"ABC Tractors Pvt Ltd",
		"Dealer: ABC Tractors Pvt Ltd",
		"Model: Mahindra 575 DI",
		"Horse Power: 50 H.P.",
		"----------------------------------------",
		"Item      Qty   Rate      Amount",
		"Tractor   1     525000    525000",
		"----------------------------------------",
		"Grand Total: Rs 5,25,000",
		"(Stamp)                              (Signature)",
	]

	for t in lines:
		page.insert_text((x, y), t, fontsize=font_size)
		y += line_h

	# Draw rectangles where a detector might find stamp/signature
	stamp = fitz.Rect(50, 700, 180, 780)
	sign = fitz.Rect(380, 700, 540, 780)
	page.draw_rect(stamp, width=1)
	page.insert_text((55, 715), "STAMP", fontsize=11)
	page.draw_rect(sign, width=1)
	page.insert_text((385, 715), "SIGN", fontsize=11)

	doc.save(str(out_path))
	doc.close()
	return out_path
