from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Box:
	x1: int
	y1: int
	x2: int
	y2: int

	def as_list(self) -> list[int]:
		return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]


def draw_boxes(
	image: Image.Image,
	labeled_boxes: Iterable[Tuple[str, Box]],
	color: tuple[int, int, int] = (255, 0, 0),
	width: int = 3,
) -> Image.Image:
	img = image.copy().convert("RGB")
	draw = ImageDraw.Draw(img)

	try:
		font = ImageFont.load_default()
	except Exception:
		font = None

	for label, b in labeled_boxes:
		draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=width)
		if label:
			xy = (b.x1 + 2, max(0, b.y1 - 12))
			draw.text(xy, label, fill=color, font=font)

	return img


def pil_to_bgr(image: Image.Image) -> np.ndarray:
	rgb = np.array(image.convert("RGB"))
	return rgb[:, :, ::-1].copy()


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
	rgb = bgr[:, :, ::-1]
	return Image.fromarray(rgb)
