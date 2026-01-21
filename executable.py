#!/usr/bin/env python3
"""Convenience wrapper.

Allows running the pipeline from repo root as:
  python executable.py ...

Internally dispatches to doc_ai_project/executable.py.
"""

from __future__ import annotations

import runpy
import sys
import os
from pathlib import Path


def main() -> None:
	repo_root = Path(__file__).resolve().parent
	doc_ai_dir = repo_root / "doc_ai_project"

	# Local-convenience default:
	# For PNG inputs, the pipeline cannot use PyMuPDF text fallback, so OCR must run.
	# Prefer Tesseract if it is installed (more stable in constrained containers).
	# Otherwise, enable PaddleOCR unless the user opts out.
	argv = sys.argv[1:]
	if ("--png" in argv) and ("--enable-paddleocr" not in argv):
		disabled = str(os.getenv("DOC_AI_DISABLE_PADDLEOCR", "0")).strip().lower() in {"1", "true", "yes"}
		enabled = str(os.getenv("DOC_AI_ENABLE_PADDLEOCR", "0")).strip().lower() in {"1", "true", "yes"}
		# If tesseract exists, don't force PaddleOCR.
		try:
			import shutil

			has_tesseract = bool(shutil.which("tesseract"))
		except Exception:
			has_tesseract = False
		if (not has_tesseract) and (not disabled) and (not enabled):
			os.environ["DOC_AI_ENABLE_PADDLEOCR"] = "1"
	# Make doc_ai_project modules importable (cost_latency, extractor, etc.)
	sys.path.insert(0, str(doc_ai_dir))
	runpy.run_path(str(doc_ai_dir / "executable.py"), run_name="__main__")


if __name__ == "__main__":
	main()
