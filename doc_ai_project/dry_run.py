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
	issues: List[str] = []

	if missing_files:
		issues.append(f"missing_submission_files: {missing_files}")

	if not pdf.exists():
		issues.append(f"pdf_missing: {pdf}")
	if not cfg.exists():
		issues.append(f"config_missing: {cfg}")

	if not issues:
		cli = _run_cli(base_dir, pdf, cfg, result_json, timeout_sec=float(args.timeout))
		runtime_ok = bool(cli.get("returncode") == 0 and float(cli.get("elapsed_sec") or 9999) < 30.0)
		if not runtime_ok:
			issues.append("runtime_or_cli_failed")

		if result_json.exists():
			try:
				obj = json.loads(result_json.read_text(encoding="utf-8"))
				InvoiceOutput.model_validate(obj)
				schema_ok = True
			except (ValidationError, Exception) as e:
				schema_ok = False
				issues.append(f"schema_validation_failed: {e}")
		else:
			issues.append("output_json_missing")

	rep: Dict[str, Any] = {
		"ok": bool((not issues) and schema_ok and runtime_ok),
		"missing_files": missing_files,
		"cli": cli,
		"schema_ok": schema_ok,
		"runtime_ok": runtime_ok,
		"result_json": str(result_json),
		"issues": issues,
	}

	out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps(rep, ensure_ascii=False, indent=2))
	return 0 if rep["ok"] else 2


if __name__ == "__main__":
	raise SystemExit(main())
