from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from utils.visualization import Box


FIELD_COLORS: Dict[str, tuple[int, int, int]] = {
	"dealer_name": (0, 102, 255),
	"model_name": (0, 153, 51),
	"horse_power": (255, 153, 0),
	"asset_cost": (204, 0, 204),
	"signature": (255, 0, 0),
	"stamp": (0, 0, 0),
}


@dataclass(frozen=True)
class ExplainArtifact:
	page_index: int
	path: Path


def save_field_overlays(
	doc_id: str,
	page_images: List[Image.Image],
	field_boxes: Dict[str, Optional[Tuple[int, int, int, int]]],
	output_dir: Path,
	field_confs: Optional[Dict[str, float]] = None,
	sig_box: Optional[Box] = None,
	stamp_box: Optional[Box] = None,
) -> List[ExplainArtifact]:
	output_dir.mkdir(parents=True, exist_ok=True)

	from PIL import ImageDraw, ImageFont

	def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> int:
		try:
			font = ImageFont.load_default()
		except Exception:
			font = None
		items = [
			("Dealer Name", FIELD_COLORS["dealer_name"]),
			("Model Name", FIELD_COLORS["model_name"]),
			("Horse Power", FIELD_COLORS["horse_power"]),
			("Asset Cost", FIELD_COLORS["asset_cost"]),
			("Signature", FIELD_COLORS["signature"]),
			("Stamp", FIELD_COLORS["stamp"]),
		]
		row_h = 14
		pad = 4
		w = 170
		h = pad * 2 + row_h * len(items)
		draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
		for i, (label, c) in enumerate(items):
			yi = y + pad + i * row_h
			draw.rectangle([x + pad, yi + 2, x + pad + 10, yi + 12], fill=c)
			draw.text((x + pad + 14, yi), label, fill=(0, 0, 0), font=font)
		return y + h + 6

	def draw_labeled_box(draw: ImageDraw.ImageDraw, label: str, b: Box, color: tuple[int, int, int]):
		try:
			font = ImageFont.load_default()
		except Exception:
			font = None
		draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=3)
		conf = None if field_confs is None else field_confs.get(label)
		cap = label
		if conf is not None:
			cap = f"{label} ({conf:.2f})"
		xy = (b.x1 + 2, max(0, b.y1 - 12))
		draw.text(xy, cap, fill=color, font=font)

	artifacts: List[ExplainArtifact] = []
	for i, img in enumerate(page_images):
		ov = img.copy().convert("RGB")
		draw = ImageDraw.Draw(ov)
		# legend
		draw_legend(draw, x=10, y=10)
		# fields
		for k, b in field_boxes.items():
			if not b:
				continue
			color = FIELD_COLORS.get(k, (255, 0, 0))
			draw_labeled_box(draw, k, Box(*b), color)
		if sig_box:
			draw_labeled_box(draw, "signature", sig_box, FIELD_COLORS["signature"])
		if stamp_box:
			draw_labeled_box(draw, "stamp", stamp_box, FIELD_COLORS["stamp"])
		p = output_dir / f"{doc_id}_page{i}_overlay.png"
		ov.save(p)
		artifacts.append(ExplainArtifact(page_index=i, path=p))

	return artifacts


def ascii_architecture_diagram() -> str:
	return (
		"PDF\n"
		"  ↓ (PyMuPDF)\n"
		"Images\n"
		"  ↓ (PaddleOCR)\n"
		"OCR Words + Boxes\n"
		"  ↓ (Line grouping / LayoutParser primitives)\n"
		"Layout Lines\n"
		"  ↓ (Rules + RapidFuzz + optional LLM hook)\n"
		"Field Candidates\n"
		"  ↓ (YOLOv8 optional)\n"
		"Signature/Stamp\n"
		"  ↓ (Validation + Postprocess)\n"
		"Strict JSON Output\n"
	)
