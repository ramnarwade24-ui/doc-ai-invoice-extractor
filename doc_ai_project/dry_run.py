from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from validator import InvoiceOutput


REQUIRED_SUBMISSION_FILES = [
	"executable.py",
	"requirements.txt",
	"README.md",
	"utils",
	"best_config.json",
]


def _check_required_files(base_dir: Path) -> List[str]:
	missing: List[str] = []
	for rel in REQUIRED_SUBMISSION_FILES:
		p = base_dir / rel
		if not p.exists():
			missing.append(rel)
	return missing


def _run_cli(base_dir: Path, pdf: Path, config: Path, out_json: Path, timeout_sec: float) -> Dict[str, Any]:
	cmd = [
		"python",
		"executable.py",
		"--config",
		str(config),
		"--pdf",
		str(pdf),
		"--out",
		str(out_json),
		"--no-eda",
		"--no-error-report",
		"--no-diagram",
	]
	start = time.perf_counter()
	p = subprocess.run(cmd, cwd=str(base_dir), capture_output=True, text=True, timeout=timeout_sec)
	elapsed = time.perf_counter() - start
	return {
		"cmd": cmd,
		"returncode": p.returncode,
		"elapsed_sec": float(round(elapsed, 3)),
		"stdout": (p.stdout or "")[-4000:],
		"stderr": (p.stderr or "")[-4000:],
	}


def _strict_schema_checks(obj: Dict[str, Any]) -> List[str]:
	"""Additional strict checks beyond pydantic model validation."""
	issues: List[str] = []
	allowed_top = {
		"doc_id",
		"fields",
		"confidence",
		"review_required",
		"processing_time_sec",
		"cost_estimate_usd",
		"cost_breakdown_usd",
	}
	extra_top = sorted(set(obj.keys()) - allowed_top)
	if extra_top:
		issues.append(f"unexpected_top_level_keys: {extra_top}")

	fields = obj.get("fields")
	if not isinstance(fields, dict):
		issues.append("fields_not_object")
		return issues

	allowed_fields = {"dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"}
	extra_fields = sorted(set(fields.keys()) - allowed_fields)
	if extra_fields:
		issues.append(f"unexpected_fields_keys: {extra_fields}")

	def _check_presence_box(name: str) -> None:
		pb = fields.get(name)
		if not isinstance(pb, dict):
			issues.append(f"{name}_not_object")
			return
		present = pb.get("present")
		bbox = pb.get("bbox")
		if not isinstance(present, bool):
			issues.append(f"{name}.present_not_bool")
			return
		if not isinstance(bbox, list) or not all(isinstance(x, int) for x in bbox):
			issues.append(f"{name}.bbox_not_int_list")
			return
		if present:
			if len(bbox) != 4:
				issues.append(f"{name}.present_true_bbox_len_not_4")
				return
			x1, y1, x2, y2 = bbox
			if any(v < 0 for v in (x1, y1, x2, y2)):
				issues.append(f"{name}.bbox_negative")
			if x2 <= x1 or y2 <= y1:
				issues.append(f"{name}.bbox_invalid_order")
		else:
			if len(bbox) != 0:
				issues.append(f"{name}.present_false_bbox_not_empty")

	_check_presence_box("signature")
	_check_presence_box("stamp")
	return issues


def main() -> int:
	p = argparse.ArgumentParser(description="Submission dry-run: CLI + schema + runtime + files")
	p.add_argument("--pdf", default="outputs/sample_invoice.pdf", help="PDF used for dry-run")
	p.add_argument("--config", default="best_config.json", help="Frozen config JSON")
	p.add_argument("--timeout", type=float, default=30.0)
	p.add_argument("--out", default="outputs/dry_run_report.json")
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	out_path = Path(args.out)
	if not out_path.is_absolute():
		out_path = base_dir / out_path
	out_path.parent.mkdir(parents=True, exist_ok=True)

	pdf = Path(args.pdf)
	if not pdf.is_absolute():
		pdf = base_dir / pdf
	cfg = Path(args.config)
	if not cfg.is_absolute():
		cfg = base_dir / cfg

	result_json = base_dir / "outputs" / "dry_run_result.json"
	result_json.parent.mkdir(parents=True, exist_ok=True)

	missing_files = _check_required_files(base_dir)
	cli = {}
	schema_ok = False
	runtime_ok = False
	cost_ok = False
	latency_ok = False
	issues: List[str] = []

	if missing_files:
		issues.append(f"missing_submission_files: {missing_files}")

	if not pdf.exists():
		issues.append(f"pdf_missing: {pdf}")
	if not cfg.exists():
		issues.append(f"config_missing: {cfg}")

	if not issues:
		cli = _run_cli(base_dir, pdf, cfg, result_json, timeout_sec=float(args.timeout))
		runtime_ok = bool(cli.get("returncode") == 0)
		if not runtime_ok:
			issues.append("runtime_or_cli_failed")

		if result_json.exists():
			try:
				obj = json.loads(result_json.read_text(encoding="utf-8"))
				InvoiceOutput.model_validate(obj, strict=True)
				issues.extend(_strict_schema_checks(obj))
				schema_ok = True
				# Enforce evaluator constraints
				latency = float(obj.get("processing_time_sec") or 9999.0)
				cost = float(obj.get("cost_estimate_usd") or 9999.0)
				latency_ok = bool(latency < 30.0)
				cost_ok = bool(cost < 0.01)
				if not latency_ok:
					issues.append(f"latency_exceeded: {latency}")
				if not cost_ok:
					issues.append(f"cost_exceeded: {cost}")
			except (ValidationError, Exception) as e:
				schema_ok = False
				issues.append(f"schema_validation_failed: {e}")
		else:
			issues.append("output_json_missing")

	rep: Dict[str, Any] = {
		"ok": bool((not issues) and schema_ok and runtime_ok and latency_ok and cost_ok),
		"missing_files": missing_files,
		"cli": cli,
		"schema_ok": schema_ok,
		"runtime_ok": runtime_ok,
		"latency_ok": latency_ok,
		"cost_ok": cost_ok,
		"result_json": str(result_json),
		"issues": issues,
	}

	out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
	# Concise evaluator summary
	if rep["ok"]:
		print("DRY-RUN: PASS")
	else:
		print("DRY-RUN: FAIL")
		for it in issues[:10]:
			print(f"- {it}")
		if len(issues) > 10:
			print(f"- ... ({len(issues) - 10} more)")
		print(f"Report: {out_path}")
	return 0 if rep["ok"] else 2


if __name__ == "__main__":
	raise SystemExit(main())
