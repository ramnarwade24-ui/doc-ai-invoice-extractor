from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import os

import numpy as np
from PIL import Image

from utils.image import preprocess_for_ocr


class OCRFailure(RuntimeError):
	pass


@dataclass(frozen=True)
class OCRWord:
	text: str
	bbox: List[List[float]]  # 4 points
	conf: float


@dataclass(frozen=True)
class OCRPage:
	page_index: int
	words: List[OCRWord]

	def texts(self) -> List[str]:
		return [w.text for w in self.words if w.text]


class PaddleOCREngine:
	def __init__(
		self,
		use_angle_cls: bool = True,
		langs: Optional[Iterable[str]] = None,
		max_retries: int = 1,
		preprocess_variants: Optional[List[tuple[bool, bool, bool]]] = None,
		autorotate: bool = False,
		adaptive_threshold: bool = False,
		shadow_remove: bool = True,
		perspective_correct: bool = False,
		upscale_if_low_res: bool = True,
	):
		# Hard-disable GPU/TensorRT/MKLDNN for safety in headless CPU-only environments.
		# These env vars must be set before importing paddle/paddleocr.
		os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
		os.environ.setdefault("FLAGS_use_cuda", "0")
		os.environ.setdefault("FLAGS_use_mkldnn", "0")
		os.environ.setdefault("FLAGS_enable_ir_optim", "0")
		try:
			from paddleocr import PaddleOCR  # type: ignore
		except Exception as e:
			raise RuntimeError(
				"PaddleOCR is not available. Install dependencies from requirements.txt"
			) from e

		# PaddleOCR language selection is model-dependent. We run a best-effort multilingual strategy:
		# try langs in order and pick the run with the best average confidence.
		self.langs = list(langs) if langs else ["en", "devanagari", "gujarati"]
		self.use_angle_cls = use_angle_cls
		self.max_retries = max(0, int(max_retries))
		self.preprocess_variants = preprocess_variants or [
			(True, True, True),
			(False, True, True),
			(True, False, True),
		]
		self.autorotate = bool(autorotate)
		self.adaptive_threshold = bool(adaptive_threshold)
		self.shadow_remove = bool(shadow_remove)
		self.perspective_correct = bool(perspective_correct)
		self.upscale_if_low_res = bool(upscale_if_low_res)
		self._ocr_by_lang = {}
		for lang in self.langs:
			try:
				self._ocr_by_lang[lang] = PaddleOCR(
					use_angle_cls=use_angle_cls,
					lang=lang,
					show_log=False,
					use_gpu=False,
					use_tensorrt=False,
					enable_mkldnn=False,
					ir_optim=False,
				)
			except Exception:
				# Unsupported lang/model; skip.
				continue
		if not self._ocr_by_lang:
			# Last-resort: english
			self._ocr_by_lang["en"] = PaddleOCR(
				use_angle_cls=use_angle_cls,
				lang="en",
				show_log=False,
				use_gpu=False,
				use_tensorrt=False,
				enable_mkldnn=False,
				ir_optim=False,
			)

	@staticmethod
	def _pil_to_np(image: Image.Image) -> np.ndarray:
		rgb = np.array(image.convert("RGB"))
		return rgb

	@staticmethod
	def _parse_result(result: Any) -> List[OCRWord]:
		words: List[OCRWord] = []
		for line in result or []:
			try:
				box, (text, conf) = line
			except Exception:
				continue
			if text is None:
				continue
			words.append(OCRWord(text=str(text), bbox=box, conf=float(conf or 0.0)))
		return words

	@staticmethod
	def _avg_conf(words: List[OCRWord]) -> float:
		if not words:
			return 0.0
		return float(sum(w.conf for w in words) / max(1, len(words)))

	def _ocr_once(self, lang: str, image: Image.Image) -> List[OCRWord]:
		ocr = self._ocr_by_lang.get(lang)
		if ocr is None:
			return []
		arr = self._pil_to_np(image)
		result: Any = ocr.ocr(arr, cls=True)
		return self._parse_result(result)

	def run_page(self, page_index: int, image: Image.Image) -> OCRPage:
		"""Run OCR with preprocessing + retry and best-lang selection."""
		attempts = []
		# Backward compatible: tuples may be length-3 (denoise, deskew, contrast)
		variants = []
		for tup in self.preprocess_variants:
			try:
				d, s, c = bool(tup[0]), bool(tup[1]), bool(tup[2])
			except Exception:
				d, s, c = True, True, True
			variants.append(
				preprocess_for_ocr(
					image,
					denoise=d,
					deskew=s,
					contrast=c,
					autorotate=self.autorotate,
					adaptive_threshold=self.adaptive_threshold,
					shadow_remove=self.shadow_remove,
					perspective_correct=self.perspective_correct,
					upscale_if_low_res=self.upscale_if_low_res,
				)
			)

		last_exc: Optional[Exception] = None
		best_overall_words: List[OCRWord] = []
		best_overall_score = 0.0
		best_overall_steps = ()

		to_try = variants[: max(1, min(len(variants), 1 + self.max_retries))]
		for v in to_try:
			best_words: List[OCRWord] = []
			best_score = 0.0
			for lang in self.langs:
				try:
					words = self._ocr_once(lang, v.image)
					score = self._avg_conf(words)
				except Exception as e:
					last_exc = e
					continue
				# Prefer higher confidence; tie-break by more text
				if (score > best_score) or (abs(score - best_score) < 1e-6 and len(words) > len(best_words)):
					best_score = score
					best_words = words

			attempts.append((v.steps, v.estimated_skew_deg, best_score, len(best_words)))
			if (best_score > best_overall_score) or (
				abs(best_score - best_overall_score) < 1e-6 and len(best_words) > len(best_overall_words)
			):
				best_overall_score = best_score
				best_overall_words = best_words
				best_overall_steps = v.steps

			# Early exit if good enough
			if best_score >= 0.70 and len(best_words) >= 8:
				return OCRPage(page_index=page_index, words=best_words)

		# Return best overall attempt (even if low) to be fail-safe
		if best_overall_words:
			return OCRPage(page_index=page_index, words=best_overall_words)

		if last_exc:
			raise OCRFailure(f"OCR failed for page {page_index}: {last_exc}") from last_exc
		return OCRPage(page_index=page_index, words=[])
