#!/usr/bin/env python3
"""Dataset preflight for jury evaluation (PNG-only).

Validates:
- `data/images/` exists and contains at least 1 PNG
- FAILS if any non-PNG invoice files exist under the invoices directory
- Deterministic discovery (recursive + sorted)

This script is intentionally lightweight and offline-safe.
It does NOT run OCR/models.

Examples:
	python dataset_preflight.py
	python dataset_preflight.py --invoices data/images --labels data/labels
	python dataset_preflight.py --invoices "data/images/**/*.png"
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



def _discover_pngs(invoices: str | Path, *, repo_root: Path) -> List[Path]:
	"""Discover PNGs deterministically.

	- If invoices is a directory: scans recursively for *.png
	- If invoices is a file: accepts a single PNG
	- Otherwise: treats as a glob pattern relative to repo root (supports **)
	"""
	inv_str = str(invoices)
	inv_path = Path(inv_str)
	resolved = inv_path if inv_path.is_absolute() else (repo_root / inv_path)

	if resolved.exists() and resolved.is_file():
		return [resolved] if resolved.suffix.lower() == ".png" else []
	if resolved.exists() and resolved.is_dir():
		return sorted([p for p in resolved.rglob("*.png") if p.is_file()])

	return sorted([p for p in repo_root.glob(inv_str) if p.is_file() and p.suffix.lower() == ".png"])



def _find_non_png_files(invoices_dir: Path) -> List[Path]:
	"""Return invoice files under invoices_dir that are NOT .png.

We ignore common non-invoice artifacts like README.md by only checking files with an extension.
"""
	bad: List[Path] = []
	for p in sorted([x for x in invoices_dir.rglob("*") if x.is_file()]):
		suf = p.suffix.lower()
		if not suf:
			continue
		if suf != ".png":
			bad.append(p)
	return bad


@dataclass(frozen=True)
class PreflightReport:
	png_count: int
	non_png_count: int
	non_png_samples: List[str]
	labels_dir: Optional[str]
	labels_present_count: Optional[int]

	def as_json(self) -> str:
		return json.dumps(
			{
				"png_count": int(self.png_count),
				"non_png_count": int(self.non_png_count),
				"non_png_samples": list(self.non_png_samples),
				"labels_dir": self.labels_dir,
				"labels_present_count": self.labels_present_count,
			},
			ensure_ascii=False,
			indent=2,
		)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Dataset preflight for data/images (PNG-only) and data/labels")
	p.add_argument(
		"--invoices",
		default="data/images",
		help=(
			"Dataset path for invoice PNGs. Accepts a directory (recursive), a single PNG, or a glob like data/images/**/*.png. "
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

	# Resolve invoices path for non-PNG checks when it's a directory
	inv_path = Path(str(args.invoices))
	inv_path = inv_path if inv_path.is_absolute() else (repo_root / inv_path)

	pngs = _discover_pngs(args.invoices, repo_root=repo_root)
	if not pngs:
		print("[FAIL] No PNGs found.")
		print("       Expected PNG invoices under data/images/.")
		return 2

	non_png: List[Path] = []
	if inv_path.exists() and inv_path.is_dir():
		non_png = _find_non_png_files(inv_path)
		if non_png:
			print("[FAIL] Non-PNG invoice files detected (PNG-only required).")
			for p in non_png[:25]:
				print(f"       - {p}")
			if len(non_png) > 25:
				print(f"       ... and {len(non_png) - 25} more")
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
		png_count=len(pngs),
		non_png_count=len(non_png),
		non_png_samples=[str(p) for p in non_png[:25]],
		labels_dir=(str(labels_dir) if labels_dir is not None else None),
		labels_present_count=labels_present_count,
	)

	out_path = Path(str(args.out))
	if not out_path.is_absolute():
		out_path = repo_root / out_path
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(report.as_json(), encoding="utf-8")

	print(f"[OK] Dataset preflight passed ({len(pngs)} PNGs).")
	if labels_present_count is None:
		print("[INFO] Labels: not found (ok for latency/cost-only checks).")
	else:
		print(f"[INFO] Labels: {labels_present_count} JSON file(s) found.")
	print(f"[OK] Wrote report: {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
