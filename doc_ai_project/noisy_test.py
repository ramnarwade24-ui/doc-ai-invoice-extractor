from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from utils.determinism import set_deterministic
from utils.pdf import pdf_to_images

try:
	import cv2  # type: ignore
except Exception:  # pragma: no cover
	cv2 = None


@dataclass(frozen=True)
class Variant:
	name: str
	kind: str
	image: Image.Image
	format: str  # "png" | "jpg"


def _load_image_or_pdf_first_page(path: Path, dpi: int) -> Image.Image:
	if path.suffix.lower() == ".pdf":
		pages = pdf_to_images(path, dpi=dpi, max_pages=1)
		if not pages:
			raise FileNotFoundError(f"No pages in PDF: {path}")
		return pages[0].image
	img = Image.open(path)
	return img.convert("RGB")


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
	a = np.array(img.convert("RGB"))
	return a[:, :, ::-1].copy()


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
	rgb = bgr[:, :, ::-1]
	return Image.fromarray(rgb)


def _rotate_small(img: Image.Image, deg: float) -> Image.Image:
	# Deterministic: fixed resample, expand and white fill
	return img.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def _gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
	return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def _motion_blur(img: Image.Image, k: int = 9) -> Image.Image:
	k = int(max(3, k | 1))  # odd
	if cv2 is None:
		# Fallback: mild gaussian blur
		return _gaussian_blur(img, radius=1.5)
	bgr = _pil_to_bgr(img)
	kernel = np.zeros((k, k), dtype=np.float32)
	kernel[k // 2, :] = 1.0
	kernel /= float(k)
	out = cv2.filter2D(bgr, -1, kernel)
	return _bgr_to_pil(out)


def _shadow_overlay(img: Image.Image, strength: float, seed: int) -> Image.Image:
	# Create a smooth gradient shadow mask
	rng = np.random.default_rng(seed)
	w, h = img.size
	# Random direction + center
	cx, cy = int(rng.integers(int(0.2 * w), int(0.8 * w))), int(rng.integers(int(0.2 * h), int(0.8 * h)))
	x = np.linspace(0, 1, w, dtype=np.float32)
	y = np.linspace(0, 1, h, dtype=np.float32)
	xv, yv = np.meshgrid(x, y)
	# Radial-ish falloff
	d = np.sqrt(((xv * w - cx) ** 2 + (yv * h - cy) ** 2))
	d = d / float(np.max(d) + 1e-6)
	mask = (1.0 - d)
	mask = np.clip(mask, 0.0, 1.0)
	mask = (1.0 - (strength * 0.85) * mask).astype(np.float32)

	arr = np.array(img.convert("RGB"), dtype=np.float32)
	arr = arr * mask[:, :, None]
	arr = np.clip(arr, 0, 255).astype(np.uint8)
	return Image.fromarray(arr)


def _brightness_contrast(img: Image.Image, brightness: float, contrast: float) -> Image.Image:
	i = ImageEnhance.Brightness(img).enhance(float(brightness))
	i = ImageEnhance.Contrast(i).enhance(float(contrast))
	return i


def _jpeg_artifacts(img: Image.Image, quality: int) -> Image.Image:
	from io import BytesIO

	buf = BytesIO()
	img.save(buf, format="JPEG", quality=int(quality), optimize=False)
	buf.seek(0)
	return Image.open(buf).convert("RGB")


def _perspective_warp(img: Image.Image, strength: float, seed: int) -> Image.Image:
	strength = float(max(0.0, min(1.0, strength)))
	if cv2 is None:
		# PIL fallback: QUAD transform
		rng = np.random.default_rng(seed)
		w, h = img.size
		dx = int(strength * 0.08 * w)
		dy = int(strength * 0.08 * h)
		quad = (
			rng.integers(0, dx),
			rng.integers(0, dy),
			w - rng.integers(0, dx),
			rng.integers(0, dy),
			w - rng.integers(0, dx),
			h - rng.integers(0, dy),
			rng.integers(0, dx),
			h - rng.integers(0, dy),
		)
		return img.transform((w, h), Image.Transform.QUAD, quad, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

	rng = np.random.default_rng(seed)
	bgr = _pil_to_bgr(img)
	h, w = bgr.shape[:2]
	max_dx = int(strength * 0.10 * w)
	max_dy = int(strength * 0.10 * h)

	src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
	dst = np.float32(
		[
			[rng.integers(0, max_dx), rng.integers(0, max_dy)],
			[w - 1 - rng.integers(0, max_dx), rng.integers(0, max_dy)],
			[w - 1 - rng.integers(0, max_dx), h - 1 - rng.integers(0, max_dy)],
			[rng.integers(0, max_dx), h - 1 - rng.integers(0, max_dy)],
		]
	)
	m = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	return _bgr_to_pil(warped)


def generate_variants(
	*,
	input_path: Path,
	out_dir: Path,
	seed: int = 1337,
	dpi: int = 200,
	profile: str = "mild",
) -> Dict[str, object]:
	"""Generate deterministic noisy variants from a clean PDF/image.

	profile:
	- mild: good for regression (should mostly pass)
	- stress: harsher distortions
	"""
	set_deterministic(seed=seed, deterministic=True)
	out_dir.mkdir(parents=True, exist_ok=True)

	img = _load_image_or_pdf_first_page(input_path, dpi=dpi)
	# Normalize to a reasonable width (deterministic)
	max_w = 1400
	if img.width > max_w:
		scale = max_w / float(img.width)
		img = img.resize((int(img.width * scale), int(img.height * scale)), resample=Image.BICUBIC)

	variants: List[Variant] = []

	# Clean baseline
	variants.append(Variant(name="clean", kind="clean", image=img, format="png"))

	if profile == "stress":
		rot = [-10, -5, 5, 10, 90]
		gauss = [1.2, 2.0]
		motion = [11]
		shadow = [0.55]
		bc = [(0.75, 1.25), (1.15, 0.85)]
		persp = [0.75]
		jpeg_q = [25]
	else:
		rot = [-5, 5, 90]
		gauss = [1.0]
		motion = [9]
		shadow = [0.35]
		bc = [(0.90, 1.10)]
		persp = [0.45]
		jpeg_q = [40]

	for d in rot:
		variants.append(Variant(name=f"rotate_{d}", kind="rotation", image=_rotate_small(img, float(d)), format="png"))
	for r in gauss:
		variants.append(Variant(name=f"blur_gauss_{r}", kind="blur", image=_gaussian_blur(img, float(r)), format="png"))
	for k in motion:
		variants.append(Variant(name=f"blur_motion_{k}", kind="blur", image=_motion_blur(img, int(k)), format="png"))
	for s in shadow:
		variants.append(
			Variant(
				name=f"shadow_{s}",
				kind="shadow",
				image=_shadow_overlay(img, float(s), seed=seed + 17),
				format="png",
			)
		)
	for b, c in bc:
		variants.append(
			Variant(
				name=f"bright_{b}_contrast_{c}",
				kind="brightness_contrast",
				image=_brightness_contrast(img, float(b), float(c)),
				format="png",
			)
		)
	for s in persp:
		variants.append(
			Variant(
				name=f"perspective_{s}",
				kind="perspective",
				image=_perspective_warp(img, float(s), seed=seed + 23),
				format="png",
			)
		)
	for q in jpeg_q:
		variants.append(
			Variant(
				name=f"jpeg_q{q}",
				kind="jpeg",
				image=_jpeg_artifacts(img, int(q)),
				format="jpg",
			)
		)

	# Write images and single-page PDFs
	manifest = {
		"input": str(input_path),
		"seed": int(seed),
		"dpi": int(dpi),
		"profile": profile,
		"variants": [],
	}

	for v in variants:
		img_path = out_dir / f"{v.name}.{v.format}"
		if v.format == "jpg":
			v.image.save(img_path, format="JPEG", quality=90)
		else:
			v.image.save(img_path)
		pdf_path = out_dir / f"{v.name}.pdf"
		# Embed image as a PDF page
		v.image.convert("RGB").save(pdf_path, format="PDF")
		manifest["variants"].append(
			{
				"name": v.name,
				"kind": v.kind,
				"image": str(img_path),
				"pdf": str(pdf_path),
			}
		)

	(out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
	return manifest


def main() -> int:
	p = argparse.ArgumentParser(description="Deterministic noisy-document generator")
	p.add_argument("--input", required=True, help="Path to clean PDF or image")
	p.add_argument("--out", default="outputs/noisy_tests", help="Output folder")
	p.add_argument("--seed", type=int, default=1337)
	p.add_argument("--dpi", type=int, default=200)
	p.add_argument("--profile", default="mild", choices=["mild", "stress"])
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	in_path = Path(args.input)
	if not in_path.is_absolute():
		in_path = base_dir / in_path
	stem = in_path.stem

	out_root = Path(args.out)
	if not out_root.is_absolute():
		out_root = base_dir / out_root
	out_dir = out_root / stem

	manifest = generate_variants(input_path=in_path, out_dir=out_dir, seed=int(args.seed), dpi=int(args.dpi), profile=str(args.profile))
	print(json.dumps({"out_dir": str(out_dir), "variants": len(manifest.get("variants", []))}, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
