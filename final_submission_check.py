#!/usr/bin/env python3
"""Final submission validation (evaluator-grade).

Checks on N random PDFs (default 10):
- Schema compliance (strict)
- Latency <= 30s
- Cost <= $0.01
- Determinism (same PDF run twice -> stable outputs, ignoring timing)
- Accuracy >= 95% (only if labels are provided)

Writes:
- doc_ai_project/outputs/final_submission_report.json

Deterministic: seeded sampling + sorted discovery.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _doc_ai_dir(repo_root: Path) -> Path:
	return repo_root / "doc_ai_project"


def _discover_pdfs(invoices: str | Path, *, repo_root: Path) -> List[Path]:
	inv_str = str(invoices)
	inv_path = Path(inv_str)
	resolved = inv_path if inv_path.is_absolute() else (repo_root / inv_path)
	if resolved.exists() and resolved.is_file():
		return [resolved] if resolved.suffix.lower() == ".pdf" else []
	if resolved.exists() and resolved.is_dir():
		return sorted([p for p in resolved.rglob("*.pdf") if p.is_file()])
	return sorted([p for p in repo_root.glob(inv_str) if p.is_file() and p.suffix.lower() == ".pdf"])


def _load_label(labels_dir: Path, doc_id: str) -> Optional[Dict[str, Any]]:
	p = labels_dir / f"{doc_id}.json"
	if not p.exists():
		return None
	try:
		obj = json.loads(p.read_text(encoding="utf-8"))
		if isinstance(obj, dict) and "fields" in obj and isinstance(obj["fields"], dict):
			return obj["fields"]
		if isinstance(obj, dict):
			return obj
		return None
	except Exception:
		return None


def _normalize_name(text: str) -> str:
	repo_root = _repo_root()
	sys.path.insert(0, str(_doc_ai_dir(repo_root)))
	utils_text = importlib.import_module("utils.text")
	return utils_text.normalize_name(text or "")


def _compare_fields(pred: Dict[str, Any], gt: Dict[str, Any]) -> bool:
	# Mirrors eval.py semantics.
	for key in ("dealer_name", "model_name"):
		if key not in gt:
			continue
		if gt.get(key) in (None, ""):
			continue
		if _normalize_name(str(pred.get(key) or "")) != _normalize_name(str(gt.get(key) or "")):
			return False

	for key in ("horse_power", "asset_cost"):
		if key not in gt:
			continue
		if gt.get(key) in (None, ""):
			continue
		try:
			if int(pred.get(key)) != int(gt.get(key)):
				return False
		except Exception:
			return False

	for key in ("signature", "stamp"):
		if key not in gt:
			continue
		g = gt.get(key)
		p = pred.get(key)
		g_present = g.get("present") if isinstance(g, dict) else g
		p_present = p.get("present") if isinstance(p, dict) else p
		if g_present is None:
			continue
		if bool(p_present) != bool(g_present):
			return False

	return True


def _strip_nondeterministic(obj: Dict[str, Any]) -> Dict[str, Any]:
	# Ignore timing variability; keep everything else.
	out = dict(obj)
	out.pop("processing_time_sec", None)
	return out


def _run_executable(*, repo_root: Path, pdf: Path, config: str, out_rel: str, timeout_sec: float) -> Tuple[int, float, str, str]:
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
	return int(p.returncode), float(round(elapsed, 3)), (p.stdout or "")[-3000:], (p.stderr or "")[-3000:]


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Final submission checker")
	p.add_argument("--invoices", default="data/pdfs", help="Dataset PDFs path (repo-root relative)")
	p.add_argument("--labels", default="", help="Optional labels folder (repo-root relative) for accuracy checks")
	p.add_argument("--config", default="best_config.json", help="Config JSON (resolved relative to doc_ai_project)")
	p.add_argument("-n", "--num", type=int, default=10, help="Number of PDFs to sample")
	p.add_argument("--seed", type=int, default=1337)
	p.add_argument("--timeout", type=float, default=30.0)
	p.add_argument("--out", default="doc_ai_project/outputs/final_submission_report.json")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = _repo_root()
	pdfs = _discover_pdfs(args.invoices, repo_root=repo_root)
	if not pdfs:
		print("[FAIL] No PDFs found.")
		return 2

	n = max(1, min(int(args.num), len(pdfs)))
	rng = random.Random(int(args.seed))
	sample = rng.sample(pdfs, k=n)

	# Schema model
	sys.path.insert(0, str(_doc_ai_dir(repo_root)))
	InvoiceOutput = importlib.import_module("validator").InvoiceOutput

	labels_dir = (repo_root / args.labels) if args.labels else None
	accuracy_available = bool(labels_dir and labels_dir.exists())

	results: List[Dict[str, Any]] = []
	ok_docs = 0
	correct_docs = 0
	deterministic_ok = 0
	latency_violations = 0
	cost_violations = 0

	for i, pdf in enumerate(sample, start=1):
		doc_id = pdf.stem
		out1 = f"final_check/{doc_id}_a.json"
		out2 = f"final_check/{doc_id}_b.json"

		case: Dict[str, Any] = {"pdf": str(pdf), "doc_id": doc_id}
		try:
			rc1, t1, so1, se1 = _run_executable(repo_root=repo_root, pdf=pdf, config=str(args.config), out_rel=out1, timeout_sec=float(args.timeout))
			rc2, t2, so2, se2 = _run_executable(repo_root=repo_root, pdf=pdf, config=str(args.config), out_rel=out2, timeout_sec=float(args.timeout))
			case["run1"] = {"rc": rc1, "elapsed_sec": t1}
			case["run2"] = {"rc": rc2, "elapsed_sec": t2}
			if rc1 != 0 or rc2 != 0:
				case["error"] = (se1 or so1 or se2 or so2)
				results.append(case)
				continue

			# Load outputs (relative --out lands in doc_ai_project/outputs)
			p1 = repo_root / "doc_ai_project" / "outputs" / out1
			p2 = repo_root / "doc_ai_project" / "outputs" / out2
			obj1 = json.loads(p1.read_text(encoding="utf-8"))
			obj2 = json.loads(p2.read_text(encoding="utf-8"))
			InvoiceOutput.model_validate(obj1, strict=True)
			InvoiceOutput.model_validate(obj2, strict=True)

			lat1 = float(obj1.get("processing_time_sec") or 0.0)
			cost1 = float(obj1.get("cost_estimate_usd") or 0.0)
			if not (lat1 <= 30.0):
				latency_violations += 1
			if not (cost1 <= 0.01):
				cost_violations += 1

			# Determinism: ignore processing_time_sec
			stable1 = _strip_nondeterministic(obj1)
			stable2 = _strip_nondeterministic(obj2)
			det_ok = bool(stable1.get("fields") == stable2.get("fields") and stable1.get("review_required") == stable2.get("review_required") and stable1.get("cost_estimate_usd") == stable2.get("cost_estimate_usd"))
			case["deterministic"] = det_ok
			if det_ok:
				deterministic_ok += 1

			# Accuracy (if labels)
			if accuracy_available and labels_dir is not None:
				gt = _load_label(labels_dir, doc_id)
				if gt is not None:
					is_correct = _compare_fields(obj1.get("fields") or {}, gt)
					case["correct"] = bool(is_correct)
					if is_correct:
						correct_docs += 1

			ok_docs += 1
			case["latency_sec"] = lat1
			case["cost_usd"] = cost1
			results.append(case)
			print(f"[{i}/{len(sample)}] OK {pdf.name} det={det_ok} lat={lat1}s cost={cost1}")
		except subprocess.TimeoutExpired:
			case["error"] = f"timeout>{args.timeout}s"
			results.append(case)
		except Exception as e:
			case["error"] = str(e)
			results.append(case)

	success_rate = float(ok_docs / max(1, len(sample)))
	det_rate = float(deterministic_ok / max(1, ok_docs)) if ok_docs else 0.0
	acc_rate = float(correct_docs / max(1, ok_docs)) if (ok_docs and accuracy_available) else None

	checks = {
		"success_rate": success_rate,
		"determinism_rate": det_rate,
		"latency_violations": int(latency_violations),
		"cost_violations": int(cost_violations),
		"accuracy_rate": acc_rate,
		"accuracy_checked": bool(accuracy_available),
	}

	ok = bool(success_rate == 1.0 and det_rate == 1.0 and latency_violations == 0 and cost_violations == 0)
	if accuracy_available and acc_rate is not None:
		ok = bool(ok and acc_rate >= 0.95)

	report = {
		"ok": ok,
		"dataset": {"invoices": str(args.invoices), "labels": str(args.labels)},
		"sample_size": int(len(sample)),
		"checks": checks,
		"results": results,
		"notes": [
			"Accuracy is only enforced when labels are provided.",
			"Determinism check ignores processing_time_sec but requires stable fields/review_required/cost.",
		],
	}

	out_path = repo_root / args.out
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

	print(json.dumps({"ok": ok, "out": str(out_path), **checks}, indent=2))
	return 0 if ok else 2


if __name__ == "__main__":
	raise SystemExit(main())
