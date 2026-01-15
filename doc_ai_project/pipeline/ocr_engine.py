from __future__ import annotations

import io
import os
import queue
import signal
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any, Dict, Optional

from PIL import Image


class OCRFailure(RuntimeError):
	"""Raised when OCR cannot be produced within constraints."""


@dataclass(frozen=True)
class OCRWord:
	text: str
	bbox: list[int]
	conf: float


@dataclass(frozen=True)
class OCRPage:
	page_index: int
	words: list[OCRWord]


def _truthy_env(name: str) -> bool:
	v = (os.getenv(name) or "").strip().lower()
	return v in {"1", "true", "yes", "y", "on"}


DOC_AI_FAST_MODE = _truthy_env("DOC_AI_FAST_MODE")


@dataclass(frozen=True)
class OCRRequest:
	type: str
	page_index: int | None = None
	image_png: bytes | None = None
	seed: int = 1337
	engine_kwargs: Dict[str, Any] | None = None


@dataclass(frozen=True)
class OCRResponse:
	ok: bool
	page_index: int
	words: list[dict]
	error: str = ""


def _serialize_page(page: OCRPage) -> list[dict]:
	out: list[dict] = []
	for w in page.words:
		out.append({"text": w.text, "bbox": w.bbox, "conf": float(w.conf)})
	return out


def _deserialize_page(page_index: int, words: list[dict]) -> OCRPage:
	ws: list[OCRWord] = []
	for w in words or []:
		try:
			ws.append(OCRWord(text=str(w.get("text") or ""), bbox=w.get("bbox") or [], conf=float(w.get("conf") or 0.0)))
		except Exception:
			continue
	return OCRPage(page_index=page_index, words=ws)


def _worker_main(req_q: Queue, resp_q: Queue) -> None:
	"""Runs inside a separate process to isolate PaddleOCR from killing the parent."""
	# Force CPU-only mode and disable common acceleration paths BEFORE importing paddle/paddleocr.
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
	os.environ.setdefault("FLAGS_use_cuda", "0")
	os.environ.setdefault("FLAGS_use_mkldnn", "0")
	os.environ.setdefault("FLAGS_enable_ir_optim", "0")
	# Import Paddle/PaddleOCR lazily inside the worker process only.
	from ocr import PaddleOCREngine  # heavy import; keep in worker
	from utils.determinism import set_deterministic

	engine: Optional[PaddleOCREngine] = None
	engine_kwargs: Dict[str, Any] = {}
	warmed_up = False

	# If PaddleOCR triggers SIGTERM inside this process, don't propagate any cleanup exceptions.
	try:
		signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
	except Exception:
		pass

	while True:
		req: OCRRequest = req_q.get()
		if req.type == "shutdown":
			return
		try:
			set_deterministic(seed=int(req.seed), deterministic=True)
			if req.type == "warmup":
				engine_kwargs = dict(req.engine_kwargs or {})
				engine = PaddleOCREngine(**engine_kwargs)
				# Minimal dummy run to trigger lazy inits without depending on external files
				dummy = Image.new("RGB", (32, 32), color=(255, 255, 255))
				_ = engine.run_page(0, dummy)
				warmed_up = True
				resp_q.put(OCRResponse(ok=True, page_index=-1, words=[]))
				continue
			if req.type == "run_page":
				# Offline-safe: never initialize PaddleOCR during timed execution unless warmup succeeded.
				if engine is None and not warmed_up:
					raise OCRFailure("paddle_models_not_warmed_up")
				if engine is None:
					engine_kwargs = dict(req.engine_kwargs or {})
					engine = PaddleOCREngine(**engine_kwargs)
				assert req.page_index is not None
				assert req.image_png is not None
				img = Image.open(io.BytesIO(req.image_png)).convert("RGB")
				page = engine.run_page(int(req.page_index), img)
				resp_q.put(OCRResponse(ok=True, page_index=int(req.page_index), words=_serialize_page(page)))
				continue
			resp_q.put(OCRResponse(ok=False, page_index=int(req.page_index or -1), words=[], error=f"unknown_request:{req.type}"))
		except BaseException as e:
			resp_q.put(OCRResponse(ok=False, page_index=int(req.page_index or -1), words=[], error=str(e)))


class OCRWorker:
	"""A singleton-ish OCR worker that runs PaddleOCR in a child process.

	This prevents PaddleOCR crashes/SIGTERM from killing the main demo/judge process.
	"""

	def __init__(self) -> None:
		self._req_q: Queue = Queue(maxsize=1)
		self._resp_q: Queue = Queue(maxsize=1)
		self._proc: Optional[Process] = None

	def _ensure_started(self) -> None:
		if self._proc is not None and self._proc.is_alive():
			return
		self._proc = Process(target=_worker_main, args=(self._req_q, self._resp_q), daemon=True)
		self._proc.start()

	def warmup(self, *, seed: int = 1337, engine_kwargs: Optional[Dict[str, Any]] = None, timeout_sec: float = 25.0) -> bool:
		if DOC_AI_FAST_MODE:
			return True
		try:
			self._ensure_started()
			self._req_q.put(OCRRequest(type="warmup", seed=int(seed), engine_kwargs=engine_kwargs or {}), timeout=1.0)
			_ = self._resp_q.get(timeout=float(timeout_sec))
			return True
		except Exception:
			# Never fail hard during warmup; demo/judge must continue.
			return False

	def run_page(self, *, page_index: int, image: Image.Image, seed: int, engine_kwargs: Optional[Dict[str, Any]], timeout_sec: float) -> OCRPage:
		if DOC_AI_FAST_MODE:
			raise OCRFailure("DOC_AI_FAST_MODE enabled")

		self._ensure_started()
		buf = io.BytesIO()
		image.convert("RGB").save(buf, format="PNG", optimize=True)
		png = buf.getvalue()

		try:
			self._req_q.put(
				OCRRequest(type="run_page", page_index=int(page_index), image_png=png, seed=int(seed), engine_kwargs=engine_kwargs or {}),
				timeout=1.0,
			)
		except Exception as e:
			raise OCRFailure(f"ocr_worker_queue_full: {e}")

		try:
			resp: OCRResponse = self._resp_q.get(timeout=float(timeout_sec))
		except queue.Empty:
			# Kill stuck worker and force fallback.
			self.reset()
			raise OCRFailure(f"paddle_ocr_timeout>{timeout_sec}s")

		if not resp.ok:
			raise OCRFailure(resp.error or "paddle_ocr_failed")
		return _deserialize_page(page_index=int(resp.page_index), words=resp.words)

	def reset(self) -> None:
		try:
			if self._proc is not None and self._proc.is_alive():
				self._proc.terminate()
				self._proc.join(timeout=1.0)
		except Exception:
			pass
		self._proc = None


_OCR_WORKER: Optional[OCRWorker] = None


def get_ocr_worker() -> OCRWorker:
	global _OCR_WORKER
	if _OCR_WORKER is None:
		_OCR_WORKER = OCRWorker()
	return _OCR_WORKER


def warmup_ocr(*, seed: int = 1337, engine_kwargs: Optional[Dict[str, Any]] = None) -> bool:
	"""Warm up PaddleOCR models once (in an isolated process).

	Safe to call multiple times.
	"""
	return get_ocr_worker().warmup(seed=int(seed), engine_kwargs=engine_kwargs or {})


def ocr_page_with_timeout(
	*,
	page_index: int,
	image: Image.Image,
	seed: int,
	engine_kwargs: Optional[Dict[str, Any]],
	timeout_sec: float = 10.0,
) -> OCRPage:
	"""Run PaddleOCR in worker process with a hard timeout."""
	return get_ocr_worker().run_page(page_index=int(page_index), image=image, seed=int(seed), engine_kwargs=engine_kwargs or {}, timeout_sec=float(timeout_sec))
