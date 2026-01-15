#!/usr/bin/env python3
"""Dataset preflight for the official folder layout.

Validates:
- `data/pdfs/` exists and contains at least 1 PDF
- PDFs can be opened with PyMuPDF (fitz) and have >= 1 page
- Deterministic discovery (recursive + sorted)

This script is intentionally lightweight and offline-safe.
It does NOT run OCR/models.

Examples:
  python dataset_preflight.py
  python dataset_preflight.py --invoices data/pdfs --labels data/labels
  python dataset_preflight.py --invoices "data/pdfs/**/*.pdf"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _discover_pdfs(invoices: str | Path, *, repo_root: Path) -> List[Path]:
	"""Discover PDFs deterministically.

	- If invoices is a directory: scans recursively for *.pdf
	- If invoices is a file: accepts a single PDF
	- Otherwise: treats as a glob pattern relative to repo root (supports **)
	"""
	inv_str = str(invoices)
	inv_path = Path(inv_str)
	resolved = inv_path if inv_path.is_absolute() else (repo_root / inv_path)

	if resolved.exists() and resolved.is_file():
		return [resolved] if resolved.suffix.lower() == ".pdf" else []
	if resolved.exists() and resolved.is_dir():
		return sorted([p for p in resolved.rglob("*.pdf") if p.is_file()])

	return sorted([p for p in repo_root.glob(inv_str) if p.is_file() and p.suffix.lower() == ".pdf"])


def _check_pdfs_openable(pdfs: List[Path]) -> None:
	try:
		import fitz  # PyMuPDF
	except Exception as e:
		raise RuntimeError(
			"PyMuPDF (fitz) is required for dataset preflight. Install doc_ai_project/requirements.txt."
		) from e

	for p in pdfs:
		try:
			doc = fitz.open(str(p))
			pages = int(doc.page_count)
			if pages <= 0:
				doc.close()
				raise ValueError("PDF has zero pages")
			# Ensure we can read at least one page stream
			doc.load_page(0)
			doc.close()
		except Exception as e:
			raise RuntimeError(f"Unreadable PDF: {p} ({e})") from e


@dataclass(frozen=True)
class PreflightReport:
	pdf_count: int
	labels_dir: Optional[str]
	labels_present_count: Optional[int]

	def as_json(self) -> str:
		return json.dumps(
			{
				"pdf_count": int(self.pdf_count),
				"labels_dir": self.labels_dir,
				"labels_present_count": self.labels_present_count,
			},
			ensure_ascii=False,
			indent=2,
		)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Dataset preflight for data/pdfs and data/labels")
	p.add_argument(
		"--invoices",
		default="data/pdfs",
		help=(
			"Dataset path for invoice PDFs. Accepts a directory (recursive), a single PDF, or a glob like data/pdfs/**/*.pdf. "
			"Relative paths are resolved from repo root."
		),
	)
	p.add_argument(
		"--labels",
		default="data/labels",
		help=(
			"Optional labels folder (same stem as PDF). Relative paths are resolved from repo root. "
			"If missing/empty, accuracy metrics will be skipped."
		),
	)
	p.add_argument("--out", default="outputs/dataset_preflight.json", help="Write JSON report here (repo-root relative)")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = _repo_root()

	pdfs = _discover_pdfs(args.invoices, repo_root=repo_root)
	if not pdfs:
		print("[FAIL] No PDFs found.")
		print("       Expected PDFs under data/pdfs/. If you have images, run: python convert_images_to_pdf.py")
		return 2

	try:
		_check_pdfs_openable(pdfs)
	except Exception as e:
		print(f"[FAIL] {e}")
		return 2

	labels_dir: Optional[Path] = None
	labels_present_count: Optional[int] = None
	if args.labels:
		cand = Path(str(args.labels))
		labels_dir = cand if cand.is_absolute() else (repo_root / cand)
		if labels_dir.exists() and labels_dir.is_dir():
			labels_present_count = len([p for p in labels_dir.glob("*.json") if p.is_file()])
		else:
			labels_dir = None
			labels_present_count = None

	report = PreflightReport(
		pdf_count=len(pdfs),
		labels_dir=(str(labels_dir) if labels_dir is not None else None),
		labels_present_count=labels_present_count,
	)

	out_path = Path(str(args.out))
	if not out_path.is_absolute():
		out_path = repo_root / out_path
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(report.as_json(), encoding="utf-8")

	print(f"[OK] Dataset preflight passed ({len(pdfs)} PDFs).")
	if labels_present_count is None:
		print("[INFO] Labels: not found (ok for latency/cost-only checks).")
	else:
		print(f"[INFO] Labels: {labels_present_count} JSON file(s) found.")
	print(f"[OK] Wrote report: {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
