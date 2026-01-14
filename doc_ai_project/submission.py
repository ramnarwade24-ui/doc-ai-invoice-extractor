from __future__ import annotations

import argparse
import json
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


def main() -> int:
	p = argparse.ArgumentParser(description="Build deterministic CPU-only submission.zip")
	p.add_argument("--pdf", default="outputs/sample_invoice.pdf", help="PDF used to validate schema")
	p.add_argument("--out", default="outputs/submission.zip", help="Output zip path")
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	outputs_dir = base_dir / "outputs"
	outputs_dir.mkdir(parents=True, exist_ok=True)

	best_cfg_path = _ensure_best_config(base_dir, outputs_dir)
	cfg = load_config_json(best_cfg_path)
	# Force deterministic + CPU-first for submission
	cfg = replace(cfg, deterministic=True, run_mode="submission")

	sample_pdf = Path(args.pdf)
	if not sample_pdf.is_absolute():
		sample_pdf = base_dir / sample_pdf
	if not sample_pdf.exists():
		raise FileNotFoundError(f"Validation PDF not found: {sample_pdf}")

	validation = _schema_validate(sample_pdf, cfg)
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
