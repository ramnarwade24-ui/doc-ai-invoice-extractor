from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def render_table_png(
	*,
	headers: Sequence[str],
	rows: Sequence[Sequence[str]],
	title: str,
	out_path: Path,
	width: int = 1100,
	row_h: int = 28,
	pad: int = 10,
) -> Path:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		font = ImageFont.load_default()
	except Exception:
		font = None

	cols = len(headers)
	table_h = (len(rows) + 1) * row_h + pad * 3 + 30
	img = Image.new("RGB", (width, table_h), (255, 255, 255))
	d = ImageDraw.Draw(img)

	d.text((pad, pad), title, fill=(20, 20, 20), font=font)
	y0 = pad + 30

	col_w = int((width - pad * 2) / max(1, cols))

	# Header background
	d.rectangle((pad, y0, width - pad, y0 + row_h), fill=(11, 95, 255))
	for i, h in enumerate(headers):
		d.text((pad + i * col_w + 6, y0 + 7), str(h), fill=(255, 255, 255), font=font)

	# Rows
	for r_i, row in enumerate(rows):
		y = y0 + row_h * (r_i + 1)
		bg = (245, 248, 255) if r_i % 2 == 0 else (255, 255, 255)
		d.rectangle((pad, y, width - pad, y + row_h), fill=bg)
		for c_i, cell in enumerate(row):
			d.text((pad + c_i * col_w + 6, y + 7), str(cell), fill=(30, 30, 30), font=font)

	# Border
	d.rectangle((pad, y0, width - pad, y0 + row_h * (len(rows) + 1)), outline=(180, 180, 180), width=1)
	img.save(out_path)
	return out_path
