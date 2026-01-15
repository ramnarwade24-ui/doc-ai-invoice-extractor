from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval import evaluate_dataset
from utils.config import PipelineConfig
from utils.determinism import set_deterministic


def _repo_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _discover_pdfs(invoices: str | Path, *, repo_root: Path) -> List[Path]:
	inv_str = str(invoices)
	inv_path = Path(inv_str)
	resolved = inv_path if inv_path.is_absolute() else (repo_root / inv_path)
	if resolved.exists() and resolved.is_file():
		return [resolved] if resolved.suffix.lower() == ".pdf" else []
	if resolved.exists() and resolved.is_dir():
		return sorted([p for p in resolved.rglob("*.pdf") if p.is_file()])
	return sorted([p for p in repo_root.glob(inv_str) if p.is_file() and p.suffix.lower() == ".pdf"])


def _limit_pdfs(pdfs: List[Path], *, limit: int | None, seed: int) -> List[Path]:
	if limit is None:
		return pdfs
	k = int(limit)
	if k <= 0:
		return []
	if k >= len(pdfs):
		return pdfs
	rng = random.Random(int(seed))
	return sorted(rng.sample(pdfs, k=k))


def _grid_region_overrides() -> List[Dict[str, Dict[str, float]]]:
	# A small, CPU-friendly grid. Add more if you have time.
	return [
		{},
		# Slightly penalize body for header-expected fields
		{"header": {"header": 1.0, "body": 0.6, "footer": 0.55, "table": 0.6, "*": 0.55}},
		# Prefer table strongly for table-expected fields
		{"table": {"table": 1.0, "body": 0.75, "header": 0.6, "footer": 0.7, "*": 0.6}},
		# Prefer footer strongly for totals
		{"footer": {"footer": 1.0, "table": 0.8, "body": 0.7, "header": 0.6, "*": 0.6}},
	]


def _grid_ocr_variants() -> List[Tuple[Tuple[bool, bool, bool], ...]]:
	return [
		((True, True, True),),
		((False, True, True),),
		((True, False, True),),
		((True, True, True), (False, True, True)),
		((True, True, True), (True, False, True)),
	]


def main() -> int:
	p = argparse.ArgumentParser(description="Auto-tune config for max DLA")
	p.add_argument(
		"--invoices",
		default="data/pdfs",
		help=(
			"Dataset path for invoice PDFs. Accepts a directory (scans recursively), a single PDF, "
			"or a glob like data/pdfs/**/*.pdf. Relative paths are resolved from repo root."
		),
	)
	p.add_argument(
		"--labels",
		default="data/labels",
		help="Folder with GT JSON (same stem as PDF). Relative paths are resolved from repo root.",
	)
	p.add_argument("--out", default="outputs/tuning_report.json", help="Output tuning report")
	p.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optional deterministic sample size (recommended for CPU-only tuning).",
	)
	p.add_argument("--seed", type=int, default=1337)
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	repo_root = _repo_root()
	outputs_dir = base_dir / "outputs"
	outputs_dir.mkdir(parents=True, exist_ok=True)
	out_path = (base_dir / args.out) if not Path(args.out).is_absolute() else Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	set_deterministic(seed=args.seed, deterministic=True)

	base_cfg = PipelineConfig(seed=args.seed, deterministic=True, run_mode="tuning")

	dealer_thresholds = [80, 85, 90, 92, 95]
	yolo_confs = [0.2, 0.25, 0.3]
	yolo_ious = [0.4, 0.5, 0.6]
	ocr_variant_sets = _grid_ocr_variants()
	region_grids = _grid_region_overrides()

	trials: List[Dict[str, Any]] = []
	best = None
	best_key = (-1.0, -1.0)  # (dla, avg_field_acc)

	pdfs = _discover_pdfs(args.invoices, repo_root=repo_root)
	pdfs = _limit_pdfs(pdfs, limit=args.limit, seed=int(args.seed))
	if not pdfs:
		raise FileNotFoundError(
			"No PDFs found. Provide --invoices as a directory, a single PDF, or a glob pattern. "
			"Example: --invoices data/pdfs"
		)
	labels_dir = Path(args.labels)
	if not labels_dir.is_absolute():
		labels_dir = repo_root / labels_dir
	if not labels_dir.exists():
		raise FileNotFoundError(
			f"Labels folder not found: {labels_dir}. Provide --labels pointing to GT JSON files." 
		)

	# Keep API stable: pass a repo-root invoices_dir plus explicit pdf list
	invoices_dir = repo_root / "data" / "pdfs"

	# Small Cartesian sweep (kept intentionally small for CPU)
	for thresh in dealer_thresholds:
		for ocr_variants in ocr_variant_sets:
			for region_overrides in region_grids:
				for yc in yolo_confs:
					for yi in yolo_ious:
						cfg = replace(
							base_cfg,
							dealer_fuzzy_threshold=int(thresh),
							yolo_conf=float(yc),
							yolo_iou=float(yi),
							ocr_preprocess_variants=tuple(ocr_variants),
							region_weight_overrides=deepcopy(region_overrides),
						)
						rep = evaluate_dataset(
							invoices_dir=invoices_dir,
							pdf_paths=pdfs,
							labels_dir=labels_dir,
							config=cfg,
							outputs_dir=outputs_dir,
						)
						sumry = rep["summary"]
						dla = float(sumry.get("dla") or 0.0)
						field_acc = sumry.get("field_accuracy") or {}
						avg_field_acc = float(sum(field_acc.values()) / max(1, len(field_acc)))
						trial = {
							"config": rep["config"],
							"summary": sumry,
						}
						trials.append(trial)
						key = (dla, avg_field_acc)
						if key > best_key:
							best_key = key
							best = trial

	report = {
		"best": best,
		"trials": trials,
		"notes": "Objective: maximize DLA, tie-break by avg field accuracy. Keep sweep small for CPU-only.",
	}
	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	(outputs_dir / "best_config.json").write_text(
		json.dumps(best["config"] if best else {}, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	print(json.dumps({"best_key": best_key, "best": best["summary"] if best else {}}, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
