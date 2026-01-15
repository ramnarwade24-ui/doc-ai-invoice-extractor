from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict

from dataclasses import replace

from executable import run_pipeline
from utils.config import PipelineConfig
from utils.config_io import load_config_json, save_config_json
from validator import InvoiceOutput


def _ensure_best_config(base_dir: Path, outputs_dir: Path) -> Path:
	cfg_path = base_dir / "best_config.json"
	if cfg_path.exists():
		return cfg_path

	# If not present, try to auto-select from tuning report
	tuning_report = outputs_dir / "tuning_report.json"
	if tuning_report.exists():
		from selector import select_best_config

		select_best_config(tuning_report=tuning_report, out_config=cfg_path)
		if cfg_path.exists():
			return cfg_path

	# Fallback: generate a default config so submission build is always one command.
	default_cfg = PipelineConfig(deterministic=True)
	save_config_json(default_cfg, cfg_path)
	return cfg_path


def _schema_validate(sample_pdf: Path, cfg: PipelineConfig) -> Dict[str, Any]:
	doc_id = sample_pdf.stem
	result = run_pipeline(pdf_path=str(sample_pdf), doc_id=doc_id, config=cfg)
	# Validate against Pydantic schema
	validated = InvoiceOutput.model_validate(result)
	return {
		"doc_id": validated.doc_id,
		"confidence": float(validated.confidence),
		"processing_time_sec": float(validated.processing_time_sec),
		"cost_estimate_usd": float(validated.cost_estimate_usd),
	}


def _zip_add(z: zipfile.ZipFile, base_dir: Path, path: Path) -> None:
	if path.is_dir():
		for p in sorted(path.rglob("*")):
			if p.is_dir():
				continue
			arc = str(p.relative_to(base_dir))
			z.write(p, arcname=arc)
	else:
		arc = str(path.relative_to(base_dir))
		z.write(path, arcname=arc)


def _sha256(path: Path) -> str:
	h = hashlib.sha256()
	with path.open("rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			h.update(chunk)
	return h.hexdigest()


def _repo_root(doc_ai_dir: Path) -> Path:
	return doc_ai_dir.parent


def _run_step(cmd: list[str], *, cwd: Path) -> None:
	print(json.dumps({"run": cmd, "cwd": str(cwd)}, ensure_ascii=False))
	subprocess.run(cmd, cwd=str(cwd), check=True)


def _finalize_pipeline(
	*,
	doc_ai_dir: Path,
	invoices: str,
	labels: str,
	seed: int,
) -> Dict[str, Any]:
	"""Run evaluator-ready finalize sequence and freeze a config."""
	repo_root = _repo_root(doc_ai_dir)
	outputs_dir = doc_ai_dir / "outputs"
	outputs_dir.mkdir(parents=True, exist_ok=True)

	# 1) Preflight (fast fail)
	_run_step(
		[
			"python",
			"preflight.py",
			"--invoices",
			invoices,
			"--config",
			"best_config.json",
			"--seed",
			str(seed),
		],
		cwd=repo_root,
	)

	# 2) Eval (produces eval_predictions + logs)
	eval_cmd = ["python", "doc_ai_project/eval.py", "--invoices", invoices, "--out", "outputs/eval_report.json", "--seed", str(seed)]
	if labels:
		eval_cmd += ["--labels", labels]
	_run_step(eval_cmd, cwd=repo_root)

	# 3) Learn adapters (uses eval_predictions; labels optional)
	adapt_cmd = ["python", "dataset_adapters.py"]
	if labels:
		adapt_cmd += ["--labels", labels]
	_run_step(adapt_cmd, cwd=repo_root)

	# 4) Confidence calibration (requires labels; best-effort)
	cal_cmd = ["python", "confidence_calibration.py", "--eval-report", "doc_ai_project/outputs/eval_report.json", "--out", "doc_ai_project/outputs/confidence_calibration.json"]
	try:
		_run_step(cal_cmd, cwd=repo_root)
	except Exception:
		# Keep going; calibration is optional if labels are missing.
		pass

	# 5) Tuning + selection (requires labels)
	if labels:
		_run_step(
			[
				"python",
				"doc_ai_project/tuning.py",
				"--invoices",
				invoices,
				"--labels",
				labels,
				"--out",
				"outputs/tuning_report.json",
				"--seed",
				str(seed),
			],
			cwd=repo_root,
		)
		_run_step(
			[
				"python",
				"doc_ai_project/selector.py",
				"--tuning-report",
				"outputs/tuning_report.json",
				"--out-config",
				"best_config.json",
				"--invoices",
				invoices,
				"--labels",
				labels,
			],
			cwd=repo_root,
		)

	# 6) Freeze final config (best_config + calibrated threshold if available)
	best_cfg_path = doc_ai_dir / "best_config.json"
	final_cfg_path = outputs_dir / "final_config.json"
	if best_cfg_path.exists():
		cfg = load_config_json(best_cfg_path)
		# Apply calibration if present
		cal_path = outputs_dir / "confidence_calibration.json"
		try:
			cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
			rec = cal.get("recommended_review_conf_threshold") if isinstance(cal, dict) else None
			if rec is not None:
				cfg = replace(cfg, review_conf_threshold=float(rec))
		except Exception:
			pass
		# Force deterministic + submission mode
		cfg = replace(cfg, deterministic=True, run_mode="submission")
		save_config_json(cfg, best_cfg_path)
		save_config_json(cfg, final_cfg_path)
	else:
		# Ensure it exists to avoid evaluator surprises
		cfg = replace(PipelineConfig(deterministic=True), run_mode="submission")
		save_config_json(cfg, best_cfg_path)
		save_config_json(cfg, final_cfg_path)

	final_hash = _sha256(final_cfg_path)
	meta = {
		"final_config_path": str(final_cfg_path),
		"final_config_sha256": final_hash,
		"dataset": {"invoices": invoices, "labels": labels},
		"seed": int(seed),
	}
	(outputs_dir / "submission_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
	return meta


def main() -> int:
	p = argparse.ArgumentParser(description="Build deterministic CPU-only submission.zip")
	p.add_argument(
		"--finalize",
		action="store_true",
		help="Run evaluator finalize pipeline (preflight/eval/tuning/selector/calibration) and freeze final config before zipping",
	)
	p.add_argument(
		"--final-check",
		action="store_true",
		help="Run final_submission_check.py gate after --finalize (schema/latency/cost/determinism; accuracy if labels provided)",
	)
	p.add_argument("--invoices", default="data/pdfs", help="Dataset PDFs path (repo-root relative) used for --finalize")
	p.add_argument("--labels", default="", help="Optional labels path (repo-root relative) used for --finalize")
	p.add_argument("--seed", type=int, default=1337)
	p.add_argument("--pdf", default="outputs/sample_invoice.pdf", help="PDF used to validate schema")
	p.add_argument("--out", default="outputs/submission.zip", help="Output zip path")
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	repo_root = _repo_root(base_dir)
	outputs_dir = base_dir / "outputs"
	outputs_dir.mkdir(parents=True, exist_ok=True)

	finalize_meta: Dict[str, Any] = {}
	if bool(args.finalize):
		finalize_meta = _finalize_pipeline(
			doc_ai_dir=base_dir,
			invoices=str(args.invoices),
			labels=str(args.labels),
			seed=int(args.seed),
		)
		if bool(args.final_check):
			# Use frozen final_config for evaluator-grade checks.
			report_path = base_dir / "outputs" / "final_submission_report.json"
			cmd = [
				"python",
				"final_submission_check.py",
				"--invoices",
				str(args.invoices),
				"--config",
				"outputs/final_config.json",
			]
			if args.labels:
				cmd += ["--labels", str(args.labels)]
			_run_step(cmd, cwd=repo_root)
			finalize_meta["final_submission_report"] = str(report_path)

	best_cfg_path = _ensure_best_config(base_dir, outputs_dir)
	cfg = load_config_json(best_cfg_path)
	# Force deterministic + CPU-first for submission
	cfg = replace(cfg, deterministic=True, run_mode="submission")

	# Schema-validation sample PDF: prefer an explicit --pdf, otherwise pick one from data/pdfs.
	sample_pdf = Path(args.pdf) if (args.pdf and str(args.pdf).strip()) else Path("")
	if sample_pdf and (not sample_pdf.is_absolute()):
		sample_pdf = base_dir / sample_pdf
	if (not sample_pdf) or (not sample_pdf.exists()):
		pdfs = list(Path("data/pdfs").glob("*.pdf"))
		if not pdfs:
			raise FileNotFoundError("No PDFs found under data/pdfs for submission packaging.")
		sample_pdf = pdfs[0]

	validation = _schema_validate(sample_pdf, cfg)
	# Lock config hash into submission metadata
	try:
		final_cfg_path = outputs_dir / "final_config.json"
		if final_cfg_path.exists():
			validation["final_config_sha256"] = _sha256(final_cfg_path)
	except Exception:
		pass
	if finalize_meta:
		validation["finalize"] = finalize_meta
	(outputs_dir / "submission_validation.json").write_text(
		json.dumps(validation, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	out_zip = Path(args.out)
	if not out_zip.is_absolute():
		out_zip = base_dir / out_zip
	out_zip.parent.mkdir(parents=True, exist_ok=True)

	include = [
		base_dir / "executable.py",
		base_dir / "requirements.txt",
		base_dir / "utils",
		base_dir / "best_config.json",
		base_dir / "README.md",
	]

	with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
		for item in include:
			_zip_add(z, base_dir, item)

	print(json.dumps({"submission_zip": str(out_zip), "schema_validation": validation}, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
