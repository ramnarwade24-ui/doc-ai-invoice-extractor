#!/usr/bin/env python3
"""Evaluator-grade preflight checks for the official dataset.

Checks:
- data/pdfs exists and is non-empty
- PDFs are readable (opens with PyMuPDF)
- Samples 3 PDFs deterministically and runs doc_ai_project/executable.py on them
- Fails fast with actionable errors

CPU-only, offline-compatible, deterministic.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _discover_pdfs(invoices: str | Path, *, repo_root: Path) -> List[Path]:
	inv_str = str(invoices)
	inv_path = Path(inv_str)
	resolved = inv_path if inv_path.is_absolute() else (repo_root / inv_path)
	if resolved.exists() and resolved.is_file():
		return [resolved] if resolved.suffix.lower() == ".pdf" else []
	if resolved.exists() and resolved.is_dir():
		return sorted([p for p in resolved.rglob("*.pdf") if p.is_file()])
	return sorted([p for p in repo_root.glob(inv_str) if p.is_file() and p.suffix.lower() == ".pdf"])


def _check_pdfs_readable(pdfs: List[Path]) -> None:
	try:
		import fitz  # PyMuPDF
	except Exception as e:
		raise RuntimeError(
			"PyMuPDF (fitz) is required to validate PDFs. Install doc_ai_project/requirements.txt."
		) from e

	for p in pdfs:
		try:
			doc = fitz.open(str(p))
			if int(doc.page_count) <= 0:
				doc.close()
				raise ValueError("PDF has zero pages")
			# load first page to ensure stream is readable
			doc.load_page(0)
			doc.close()
		except Exception as e:
			raise RuntimeError(f"Unreadable PDF: {p} ({e})") from e


@dataclass(frozen=True)
class ExecRun:
	pdf: str
	returncode: int
	elapsed_sec: float
	stdout_tail: str
	stderr_tail: str


def _run_executable(*, repo_root: Path, pdf: Path, config: str, out_rel: str, timeout_sec: float) -> ExecRun:
	cmd = [
		"python",
		"doc_ai_project/executable.py",
		"--config",
		config,
		"--pdf",
		str(pdf.relative_to(repo_root) if pdf.is_relative_to(repo_root) else pdf),
		"--out",
		out_rel,
		"--no-eda",
		"--no-error-report",
		"--no-diagram",
	]
	start = time.perf_counter()
	p = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=timeout_sec)
	elapsed = time.perf_counter() - start
	return ExecRun(
		pdf=str(pdf),
		returncode=int(p.returncode),
		elapsed_sec=float(round(elapsed, 3)),
		stdout_tail=(p.stdout or "")[-4000:],
		stderr_tail=(p.stderr or "")[-4000:],
	)


def _load_and_validate_output(repo_root: Path, out_rel: str) -> Dict[str, Any]:
	# executable.py writes under doc_ai_project/outputs/ when given a relative --out
	out_path = repo_root / "doc_ai_project" / "outputs" / out_rel
	if not out_path.exists():
		raise FileNotFoundError(f"Missing output JSON: {out_path}")

	obj = json.loads(out_path.read_text(encoding="utf-8"))
	# Strict-ish validation using the project schema
	sys.path.insert(0, str(repo_root / "doc_ai_project"))
	import importlib
	from pydantic import ValidationError

	InvoiceOutput = importlib.import_module("validator").InvoiceOutput
	try:
		InvoiceOutput.model_validate(obj, strict=True)
	except ValidationError as e:
		raise RuntimeError(f"Schema validation failed for {out_path}: {e}") from e
	return obj


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Preflight checks for data/pdfs dataset")
	p.add_argument(
		"--invoices",
		default="data/pdfs",
		help=(
			"Dataset path for invoice PDFs. Accepts a directory (recursive), a single PDF, or a glob like data/pdfs/**/*.pdf."
		),
	)
	p.add_argument(
		"--config",
		default="best_config.json",
		help=(
			"Config JSON to use when running executable.py (resolved relative to doc_ai_project). "
			"Default: best_config.json"
		),
	)
	p.add_argument("--samples", type=int, default=3, help="Number of PDFs to sample and run")
	p.add_argument("--seed", type=int, default=1337, help="Deterministic sampling seed")
	p.add_argument("--timeout", type=float, default=30.0, help="Per-document timeout seconds")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = _repo_root()

	pdfs = _discover_pdfs(args.invoices, repo_root=repo_root)
	if not pdfs:
		print("[FAIL] No PDFs found.")
		print("       Expected PDFs under data/pdfs/. Run: python convert_images_to_pdf.py")
		return 2

	print(f"[INFO] Found {len(pdfs)} PDF(s). Checking readability...")
	try:
		_check_pdfs_readable(pdfs)
	except Exception as e:
		print(f"[FAIL] {e}")
		return 2
	print("[OK] All PDFs opened successfully.")

	n = max(1, int(args.samples))
	n = min(n, len(pdfs))
	rng = random.Random(int(args.seed))
	samples = rng.sample(pdfs, k=n) if len(pdfs) >= n else list(pdfs)

	print(f"[INFO] Running executable.py on {len(samples)} sampled PDF(s)...")
	for pdf in samples:
		out_rel = f"preflight/{pdf.stem}.json"
		try:
			run = _run_executable(
				repo_root=repo_root,
				pdf=pdf,
				config=str(args.config),
				out_rel=out_rel,
				timeout_sec=float(args.timeout),
			)
			if run.returncode != 0:
				print(f"[FAIL] executable.py failed on {pdf} (rc={run.returncode}, {run.elapsed_sec}s)")
				print("       stderr tail:")
				print(run.stderr_tail)
				return 2
			obj = _load_and_validate_output(repo_root, out_rel)
			print(
				f"[OK] {Path(pdf).name}: latency={obj.get('processing_time_sec')}s cost=${obj.get('cost_estimate_usd')}"
			)
		except subprocess.TimeoutExpired:
			print(f"[FAIL] Timeout running executable.py on {pdf} (> {args.timeout}s)")
			return 2
		except Exception as e:
			print(f"[FAIL] {e}")
			return 2

	print("[PASS] Preflight checks succeeded.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
