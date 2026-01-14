from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

try:
	import cv2  # type: ignore
except Exception:  # pragma: no cover
	cv2 = None
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessResult:
	image: Image.Image
	steps: Tuple[str, ...]
	estimated_skew_deg: float


def pil_to_bgr(image: Image.Image) -> np.ndarray:
	rgb = np.array(image.convert("RGB"))
	return rgb[:, :, ::-1].copy()


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
	rgb = bgr[:, :, ::-1]
	return Image.fromarray(rgb)


def _estimate_skew_deg(gray: np.ndarray) -> float:
	"""Estimate skew angle in degrees via minAreaRect on foreground pixels."""
	if cv2 is None:
		return 0.0
	# Binarize
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	_, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# Invert to get text as foreground
	inv = 255 - bw
	coords = cv2.findNonZero(inv)
	if coords is None or len(coords) < 200:
		return 0.0
	rect = cv2.minAreaRect(coords)
	angle = rect[-1]
	# OpenCV angle conventions
	if angle < -45:
		angle = 90 + angle
	return float(angle)


def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
	if cv2 is None:
		return image
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
	return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _rotate_90(image: np.ndarray, k: int) -> np.ndarray:
	if cv2 is None:
		return image
	k = int(k) % 4
	if k == 0:
		return image
	if k == 1:
		return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	if k == 2:
		return cv2.rotate(image, cv2.ROTATE_180)
	return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _illumination_normalize(gray: np.ndarray) -> np.ndarray:
	"""Reduce shadows/faded ink by dividing by a blurred background."""
	if cv2 is None:
		return gray
	bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
	# Avoid division artifacts
	bg = np.maximum(bg, 1)
	norm = cv2.divide(gray, bg, scale=255)
	return norm


def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
	if cv2 is None:
		return gray
	# Adaptive threshold tends to help on uneven lighting
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	bw = cv2.adaptiveThreshold(
		blur,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		blockSize=31,
		C=10,
	)
	return bw


def _maybe_perspective_correct(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
	"""Best-effort document perspective correction.

	Returns (image, applied?). Skips if no strong 4-corner contour found.
	"""
	if cv2 is None:
		return bgr, False
	try:
		gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
		gray = _illumination_normalize(gray)
		edges = cv2.Canny(gray, 50, 150)
		edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
		cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not cnts:
			return bgr, False
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
		best = None
		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			if len(approx) == 4:
				best = approx
				break
		if best is None:
			return bgr, False
		pts = best.reshape(4, 2).astype("float32")

		# order points: tl, tr, br, bl
		s = pts.sum(axis=1)
		diff = np.diff(pts, axis=1)
		tl = pts[np.argmin(s)]
		br = pts[np.argmax(s)]
		tr = pts[np.argmin(diff)]
		bl = pts[np.argmax(diff)]
		ordered = np.array([tl, tr, br, bl], dtype="float32")

		w1 = np.linalg.norm(br - bl)
		w2 = np.linalg.norm(tr - tl)
		h1 = np.linalg.norm(tr - br)
		h2 = np.linalg.norm(tl - bl)
		w = int(max(w1, w2))
		h = int(max(h1, h2))
		if w < 400 or h < 400:
			return bgr, False

		dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
		m = cv2.getPerspectiveTransform(ordered, dst)
		warped = cv2.warpPerspective(bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
		return warped, True
	except Exception:
		return bgr, False


def _autorotate_coarse(bgr: np.ndarray) -> tuple[np.ndarray, int, float]:
	"""Choose among 0/90/180/270 degrees to maximize text-likeness.

	Score uses skew closeness to 0 + foreground density (no OCR needed).
	Returns (image, rot_k, best_score).
	"""
	if cv2 is None:
		return bgr, 0, 0.0
	best = (bgr, 0, -1e9)
	for k in (0, 1, 2, 3):
		cand = _rotate_90(bgr, k)
		gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
		gray = _illumination_normalize(gray)
		blur = cv2.GaussianBlur(gray, (3, 3), 0)
		_, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		inv = 255 - bw
		fg = float(np.count_nonzero(inv)) / float(inv.size)
		skew = abs(_estimate_skew_deg(gray))
		score = (fg * 2.0) - (skew / 20.0)
		if score > best[2]:
			best = (cand, k, score)
	return best


def preprocess_for_ocr(
	image: Image.Image,
	denoise: bool = True,
	deskew: bool = True,
	contrast: bool = True,
	autorotate: bool = False,
	adaptive_threshold: bool = False,
	shadow_remove: bool = True,
	perspective_correct: bool = False,
	upscale_if_low_res: bool = True,
) -> PreprocessResult:
	"""Lightweight preprocessing optimized for CPU-only inference."""
	if cv2 is None:
		return PreprocessResult(image=image, steps=("skip_no_cv2",), estimated_skew_deg=0.0)
	bgr = pil_to_bgr(image)
	steps = []

	# Upscale low-res photos (helps OCR on camera captures)
	if upscale_if_low_res:
		h, w = bgr.shape[:2]
		min_side = min(h, w)
		if min_side < 900:
			scale = 2.0 if min_side < 600 else 1.5
			bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
			steps.append("upscale")

	if autorotate:
		bgr, k, _ = _autorotate_coarse(bgr)
		if k != 0:
			steps.append(f"autorotate_{k * 90}")

	if perspective_correct:
		bgr2, applied = _maybe_perspective_correct(bgr)
		if applied:
			bgr = bgr2
			steps.append("perspective")

	if denoise:
		# Preserves edges while removing noise
		bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)
		steps.append("denoise")

	skew = 0.0
	if deskew:
		gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
		if shadow_remove:
			gray = _illumination_normalize(gray)
			steps.append("shadow_norm")
		skew = _estimate_skew_deg(gray)
		# Apply only for meaningful skew
		if abs(skew) >= 1.0 and abs(skew) <= 12.0:
			bgr = _rotate(bgr, skew)
			steps.append("deskew")

	if contrast:
		lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		cl = clahe.apply(l)
		merged = cv2.merge((cl, a, b))
		bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
		steps.append("clahe")

	if adaptive_threshold:
		gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
		if shadow_remove and "shadow_norm" not in steps:
			gray = _illumination_normalize(gray)
			steps.append("shadow_norm")
		bw = _adaptive_binarize(gray)
		bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
		steps.append("adaptive_thresh")

	return PreprocessResult(image=bgr_to_pil(bgr), steps=tuple(steps), estimated_skew_deg=skew)
