from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ChartStyle:
	width: int = 900
	height: int = 420
	margin: int = 50
	bg: Tuple[int, int, int] = (255, 255, 255)
	fg: Tuple[int, int, int] = (30, 30, 30)
	bar: Tuple[int, int, int] = (11, 95, 255)
	grid: Tuple[int, int, int] = (220, 220, 220)


def _font(size: int = 14):
	try:
		return ImageFont.load_default()
	except Exception:
		return None


def bar_chart_png(
	*,
	title: str,
	labels: Sequence[str],
	values: Sequence[float],
	out_path: Path,
	style: ChartStyle = ChartStyle(),
	value_range: Tuple[float, float] | None = None,
) -> Path:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	img = Image.new("RGB", (style.width, style.height), style.bg)
	d = ImageDraw.Draw(img)
	font = _font()

	# Title
	d.text((style.margin, 10), title, fill=style.fg, font=font)

	if not labels:
		img.save(out_path)
		return out_path

	vmin = min(values) if values else 0.0
	vmax = max(values) if values else 1.0
	if value_range is not None:
		vmin, vmax = value_range
	if vmax <= vmin:
		vmax = vmin + 1.0

	left = style.margin
	top = 40
	right = style.width - style.margin
	bottom = style.height - style.margin

	# Axes
	d.line((left, top, left, bottom), fill=style.fg, width=2)
	d.line((left, bottom, right, bottom), fill=style.fg, width=2)

	# Grid
	for i in range(1, 5):
		y = top + int((bottom - top) * i / 5)
		d.line((left, y, right, y), fill=style.grid, width=1)

	# Bars
	n = len(labels)
	gap = 8
	bar_w = max(6, int((right - left - gap * (n + 1)) / n))
	for i, (lab, val) in enumerate(zip(labels, values)):
		x1 = left + gap + i * (bar_w + gap)
		x2 = x1 + bar_w
		h = int((bottom - top) * (float(val) - vmin) / (vmax - vmin))
		y1 = bottom - h
		d.rectangle((x1, y1, x2, bottom), fill=style.bar)
		# label (rotated-ish by stacking)
		lab_txt = str(lab)[:14]
		d.text((x1, bottom + 4), lab_txt, fill=style.fg, font=font)

	img.save(out_path)
	return out_path


def histogram_png(
	*,
	title: str,
	values: Iterable[float],
	bins: int,
	out_path: Path,
	style: ChartStyle = ChartStyle(),
) -> Path:
	vals = [float(v) for v in values if v is not None]
	out_path.parent.mkdir(parents=True, exist_ok=True)
	img = Image.new("RGB", (style.width, style.height), style.bg)
	d = ImageDraw.Draw(img)
	font = _font()
	d.text((style.margin, 10), title, fill=style.fg, font=font)

	if not vals:
		img.save(out_path)
		return out_path

	vmin, vmax = min(vals), max(vals)
	if vmax <= vmin:
		vmax = vmin + 1.0

	counts = [0] * bins
	for v in vals:
		idx = int((v - vmin) / (vmax - vmin) * (bins - 1))
		counts[max(0, min(bins - 1, idx))] += 1

	left = style.margin
	top = 40
	right = style.width - style.margin
	bottom = style.height - style.margin

	d.line((left, top, left, bottom), fill=style.fg, width=2)
	d.line((left, bottom, right, bottom), fill=style.fg, width=2)

	max_c = max(counts) if counts else 1
	gap = 2
	bar_w = max(2, int((right - left - gap * (bins + 1)) / bins))
	for i, c in enumerate(counts):
		x1 = left + gap + i * (bar_w + gap)
		x2 = x1 + bar_w
		h = int((bottom - top) * (c / max_c))
		y1 = bottom - h
		d.rectangle((x1, y1, x2, bottom), fill=style.bar)

	img.save(out_path)
	return out_path
