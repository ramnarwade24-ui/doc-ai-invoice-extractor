#!/usr/bin/env python3
"""Batch smoke test for the official dataset (data/pdfs).

Runs doc_ai_project/executable.py on N randomly sampled PDFs (deterministic seed),
reports success rate + latency stats, and writes outputs/smoke_report.json.

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
from typing import Any, Dict, List


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


@dataclass(frozen=True)
class CaseResult:
	pdf: str
	ok: bool
	returncode: int
	elapsed_sec: float
	processing_time_sec: float | None
	cost_estimate_usd: float | None
	error: str


def _run_case(*, repo_root: Path, pdf: Path, config: str, timeout_sec: float) -> CaseResult:
	out_rel = f"smoke/{pdf.stem}.json"
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
	try:
		p = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=timeout_sec)
	except subprocess.TimeoutExpired:
		elapsed = time.perf_counter() - start
		return CaseResult(
			pdf=str(pdf),
			ok=False,
			returncode=124,
			elapsed_sec=float(round(elapsed, 3)),
			processing_time_sec=None,
			cost_estimate_usd=None,
			error=f"timeout>{timeout_sec}s",
		)

	elapsed = time.perf_counter() - start
	if p.returncode != 0:
		return CaseResult(
			pdf=str(pdf),
			ok=False,
			returncode=int(p.returncode),
			elapsed_sec=float(round(elapsed, 3)),
			processing_time_sec=None,
			cost_estimate_usd=None,
			error=(p.stderr or p.stdout or "").strip()[-4000:],
		)

	# Validate output JSON with project schema
	out_path = repo_root / "doc_ai_project" / "outputs" / out_rel
	try:
		obj = json.loads(out_path.read_text(encoding="utf-8"))
	except Exception as e:
		return CaseResult(
			pdf=str(pdf),
			ok=False,
			returncode=int(p.returncode),
			elapsed_sec=float(round(elapsed, 3)),
			processing_time_sec=None,
			cost_estimate_usd=None,
			error=f"missing_or_invalid_output_json: {out_path} ({e})",
		)

	sys.path.insert(0, str(repo_root / "doc_ai_project"))
	import importlib
	from pydantic import ValidationError

	InvoiceOutput = importlib.import_module("validator").InvoiceOutput

	try:
		InvoiceOutput.model_validate(obj, strict=True)
	except ValidationError as e:
		return CaseResult(
			pdf=str(pdf),
			ok=False,
			returncode=int(p.returncode),
			elapsed_sec=float(round(elapsed, 3)),
			processing_time_sec=None,
			cost_estimate_usd=None,
			error=f"schema_validation_failed: {e}",
		)

	return CaseResult(
		pdf=str(pdf),
		ok=True,
		returncode=int(p.returncode),
		elapsed_sec=float(round(elapsed, 3)),
		processing_time_sec=float(obj.get("processing_time_sec")) if obj.get("processing_time_sec") is not None else None,
		cost_estimate_usd=float(obj.get("cost_estimate_usd")) if obj.get("cost_estimate_usd") is not None else None,
		error="",
	)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Run a deterministic smoke test on a PDFs dataset")
	p.add_argument(
		"--invoices",
		default="data/pdfs",
		help="Dataset path for PDFs (dir recursive, single PDF, or glob like data/pdfs/**/*.pdf)",
	)
	p.add_argument("-n", "--num", "--n", type=int, default=10, help="Number of PDFs to sample")
	p.add_argument("--seed", type=int, default=1337, help="Deterministic sampling seed")
	p.add_argument(
		"--config",
		default="best_config.json",
		help="Config JSON to use (resolved relative to doc_ai_project). Default: best_config.json",
	)
	p.add_argument("--timeout", type=float, default=30.0, help="Per-document timeout seconds")
	p.add_argument("--out", default="outputs/smoke_report.json", help="Output report path (repo-root relative)")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = _repo_root()

	pdfs = _discover_pdfs(args.invoices, repo_root=repo_root)
	if not pdfs:
		print("[FAIL] No PDFs found. Expected data/pdfs/. Run: python convert_images_to_pdf.py")
		return 2

	n = max(1, int(args.num))
	n = min(n, len(pdfs))
	rng = random.Random(int(args.seed))
	sample = rng.sample(pdfs, k=n) if len(pdfs) >= n else list(pdfs)

	print(f"[INFO] Smoke-testing {len(sample)}/{len(pdfs)} PDF(s)...")
	results: List[Dict[str, Any]] = []
	ok_count = 0
	latencies: List[float] = []
	crashes: List[Dict[str, Any]] = []

	for i, pdf in enumerate(sample, start=1):
		case = _run_case(repo_root=repo_root, pdf=pdf, config=str(args.config), timeout_sec=float(args.timeout))
		results.append(case.__dict__)
		status = "OK" if case.ok else "FAIL"
		print(f"[{i}/{len(sample)}] {status} {Path(pdf).name} ({case.elapsed_sec}s)")
		if case.ok:
			ok_count += 1
			if case.processing_time_sec is not None:
				latencies.append(float(case.processing_time_sec))
		else:
			crashes.append({"pdf": case.pdf, "error": case.error, "returncode": case.returncode})

	success_rate = float(ok_count / max(1, len(sample)))
	avg_latency = float(sum(latencies) / max(1, len(latencies))) if latencies else 0.0

	report: Dict[str, Any] = {
		"dataset": str(args.invoices),
		"sample_size": int(len(sample)),
		"total_pdfs_found": int(len(pdfs)),
		"success_rate": success_rate,
		"avg_latency_sec": avg_latency,
		"crashes": crashes,
		"results": results,
	}

	out_path = Path(args.out)
	if not out_path.is_absolute():
		out_path = repo_root / out_path
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

	print(json.dumps({"success_rate": success_rate, "avg_latency_sec": avg_latency, "report": str(out_path)}, indent=2))
	return 0 if ok_count == len(sample) else 2


if __name__ == "__main__":
	raise SystemExit(main())
