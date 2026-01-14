from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from utils.visualization import Box, pil_to_bgr


@dataclass(frozen=True)
class Detection:
	label: str
	bbox: Box
	conf: float


class SignatureStampDetector:
	"""YOLOv8-based detector if weights are provided.

	If no weights are configured, returns no detections (graceful fallback).
	"""

	def __init__(
		self,
		weights_path: Optional[Path],
		conf: float = 0.25,
		iou_thresh: float = 0.5,
		img_sizes: Optional[tuple[int, ...]] = None,
	):
		self.weights_path = weights_path
		self.conf = conf
		self.iou_thresh = float(iou_thresh)
		self.img_sizes = tuple(img_sizes) if img_sizes else (640,)
		self._model = None
		# Accept alternate class names
		self.signature_aliases = {"signature", "sign"}
		self.stamp_aliases = {"stamp", "seal"}

	def _load(self):
		if self._model is not None:
			return
		if not self.weights_path:
			self._model = False
			return
		if not Path(self.weights_path).exists():
			self._model = False
			return
		try:
			from ultralytics import YOLO  # type: ignore
		except Exception:
			self._model = False
			return
		# ultralytics runs on CPU by default when no GPU is present
		self._model = YOLO(str(self.weights_path))

	@staticmethod
	def _iou(a: Box, b: Box) -> float:
		ix1 = max(a.x1, b.x1)
		iy1 = max(a.y1, b.y1)
		ix2 = min(a.x2, b.x2)
		iy2 = min(a.y2, b.y2)
		if ix2 <= ix1 or iy2 <= iy1:
			return 0.0
		inter = float((ix2 - ix1) * (iy2 - iy1))
		area_a = float(max(1, (a.x2 - a.x1) * (a.y2 - a.y1)))
		area_b = float(max(1, (b.x2 - b.x1) * (b.y2 - b.y1)))
		return inter / (area_a + area_b - inter)

	def _normalize_label(self, label: str) -> str:
		l = (label or "").strip().lower()
		if l in self.signature_aliases:
			return "signature"
		if l in self.stamp_aliases:
			return "stamp"
		return l

	def _nms(self, dets: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
		"""Greedy NMS per normalized label."""
		out: List[Detection] = []
		by_label: Dict[str, List[Detection]] = {}
		for d in dets:
			by_label.setdefault(d.label, []).append(d)

		for label, items in by_label.items():
			items.sort(key=lambda d: d.conf, reverse=True)
			kept: List[Detection] = []
			for d in items:
				suppress = False
				for k in kept:
					if self._iou(d.bbox, k.bbox) >= iou_thresh:
						suppress = True
						break
				if not suppress:
					kept.append(d)
			out.extend(kept)
		return out

	def detect(self, image: Image.Image) -> List[Detection]:
		self._load()
		if not self._model or self._model is False:
			return []

		bgr = pil_to_bgr(image)
		out: List[Detection] = []
		# Multi-scale inference: run YOLO at different img sizes and merge
		for sz in self.img_sizes:
			results = self._model.predict(bgr, conf=self.conf, imgsz=int(sz), device="cpu", verbose=False)
			if not results:
				continue
			for r in results:
				if r.boxes is None:
					continue
				boxes = r.boxes
				xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
				confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
				clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
				names: Dict[int, str] = getattr(self._model.model, "names", {}) or getattr(self._model, "names", {}) or {}

				for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
					label = self._normalize_label(names.get(int(k), str(int(k))))
					out.append(
						Detection(
							label=label,
							bbox=Box(int(x1), int(y1), int(x2), int(y2)),
							conf=float(c),
						)
					)

		# Apply NMS with configurable IoU
		out = self._nms(out, iou_thresh=self.iou_thresh)
		# Keep only target classes
		out = [d for d in out if d.label in {"signature", "stamp"}]
		return out


def best_by_label(detections: List[Detection], label: str) -> Optional[Detection]:
	cand = [d for d in detections if d.label.lower() == label.lower()]
	if not cand:
		return None
	cand.sort(key=lambda d: d.conf, reverse=True)
	return cand[0]
