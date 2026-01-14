from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from executable import run_pipeline
from noisy_test import generate_variants
from utils.config import PipelineConfig
from utils.determinism import set_deterministic
from utils.table_viz import render_table_png
from utils.text import normalize_name


FIELDS = ["dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"]


def _field_value(obj: Dict[str, Any], field: str) -> Any:
	fields = obj.get("fields") or {}
	return fields.get(field)


def _present_bool(v: Any) -> bool | None:
	if v is None:
		return None
	if isinstance(v, bool):
		return v
	if isinstance(v, dict):
		pv = v.get("present")
		if isinstance(pv, bool):
			return pv
		return None
	if isinstance(v, (int, float)):
		return bool(v)
	if isinstance(v, str):
		t = v.strip().lower()
		if t in {"true", "yes", "y", "1"}:
			return True
		if t in {"false", "no", "n", "0"}:
			return False
	return None


def _match_field(clean: Dict[str, Any], noisy: Dict[str, Any], field: str, *, hp_tol: int, cost_tol: int) -> bool:
	cv = _field_value(clean, field)
	nv = _field_value(noisy, field)
	if field in {"dealer_name", "model_name"}:
		return normalize_name(str(cv or "")) == normalize_name(str(nv or ""))
	if field == "horse_power":
		try:
			if cv is None or nv is None:
				return False
			return abs(int(cv) - int(nv)) <= int(hp_tol)
		except Exception:
			return False
	if field == "asset_cost":
		try:
			if cv is None or nv is None:
				return False
			return abs(int(cv) - int(nv)) <= int(cost_tol)
		except Exception:
			return False
	if field in {"signature", "stamp"}:
		return _present_bool(cv) == _present_bool(nv)
	return False


def main() -> int:
	p = argparse.ArgumentParser(description="Robustness evaluation on noisy variants")
	p.add_argument("--input", required=True, help="Clean PDF or image")
	p.add_argument("--profile", default="mild", choices=["mild", "stress"])
	p.add_argument("--seed", type=int, default=1337)
	p.add_argument("--dpi", type=int, default=200)
	p.add_argument("--hp-tol", type=int, default=2)
	p.add_argument("--cost-tol", type=int, default=1000)
	p.add_argument("--out", default="outputs/robustness_report.json")
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	in_path = Path(args.input)
	if not in_path.is_absolute():
		in_path = base_dir / in_path

	outputs_dir = base_dir / "outputs"
	outputs_dir.mkdir(parents=True, exist_ok=True)
	noisy_dir = outputs_dir / "noisy_tests" / in_path.stem

	set_deterministic(seed=int(args.seed), deterministic=True)

	manifest = generate_variants(
		input_path=in_path,
		out_dir=noisy_dir,
		seed=int(args.seed),
		dpi=int(args.dpi),
		profile=str(args.profile),
	)

	cfg = PipelineConfig(dpi=int(args.dpi), deterministic=True, seed=int(args.seed), run_mode="normal")

	# Run pipeline on clean and each variant
	variants = manifest.get("variants") or []
	by_name = {v["name"]: v for v in variants}
	if "clean" not in by_name:
		raise RuntimeError("manifest missing clean variant")

	clean_pdf = Path(by_name["clean"]["pdf"])
	clean_out = run_pipeline(pdf_path=str(clean_pdf), doc_id=f"{in_path.stem}_clean", config=cfg)

	rows: List[Dict[str, Any]] = []
	acc_drop_by_kind: Dict[str, List[float]] = defaultdict(list)
	conf_drop_by_kind: Dict[str, List[float]] = defaultdict(list)
	lat_delta_by_kind: Dict[str, List[float]] = defaultdict(list)

	clean_conf = float(clean_out.get("confidence") or 0.0)
	clean_lat = float(clean_out.get("processing_time_sec") or 0.0)

	for v in variants:
		name = v["name"]
		kind = v.get("kind") or "unknown"
		pdf = Path(v["pdf"])
		out = run_pipeline(pdf_path=str(pdf), doc_id=f"{in_path.stem}_{name}", config=cfg)
		matches = {f: _match_field(clean_out, out, f, hp_tol=int(args.hp_tol), cost_tol=int(args.cost_tol)) for f in FIELDS}
		match_rate = float(sum(1 for ok in matches.values() if ok) / float(len(FIELDS)))
		conf = float(out.get("confidence") or 0.0)
		lat = float(out.get("processing_time_sec") or 0.0)
		conf_drop = float(clean_conf - conf)
		lat_delta = float(lat - clean_lat)

		rows.append(
			{
				"name": name,
				"kind": kind,
				"pdf": str(pdf),
				"match_rate": match_rate,
				"matches": matches,
				"confidence": conf,
				"confidence_drop": conf_drop,
				"latency_sec": lat,
				"latency_delta_sec": lat_delta,
				"review_required": bool(out.get("review_required")),
			}
		)

		if name != "clean":
			acc_drop_by_kind[kind].append(float(1.0 - match_rate))
			conf_drop_by_kind[kind].append(float(conf_drop))
			lat_delta_by_kind[kind].append(float(lat_delta))

	# Summaries per kind
	def _avg(xs: List[float]) -> float:
		return float(sum(xs) / max(1, len(xs))) if xs else 0.0

	per_kind = {}
	for kind in sorted(set(list(acc_drop_by_kind.keys()) + list(conf_drop_by_kind.keys()) + list(lat_delta_by_kind.keys()))):
		per_kind[kind] = {
			"avg_accuracy_drop": _avg(acc_drop_by_kind.get(kind, [])),
			"avg_confidence_drop": _avg(conf_drop_by_kind.get(kind, [])),
			"avg_latency_delta_sec": _avg(lat_delta_by_kind.get(kind, [])),
			"n": int(len(acc_drop_by_kind.get(kind, [])) or len(conf_drop_by_kind.get(kind, [])) or len(lat_delta_by_kind.get(kind, []))),
		}

	report = {
		"input": str(in_path),
		"profile": str(args.profile),
		"seed": int(args.seed),
		"dpi": int(args.dpi),
		"clean": {
			"pdf": str(clean_pdf),
			"confidence": clean_conf,
			"latency_sec": clean_lat,
			"review_required": bool(clean_out.get("review_required")),
			"fields": clean_out.get("fields") or {},
		},
		"rows": rows,
		"per_kind": per_kind,
		"tolerances": {"hp": int(args.hp_tol), "cost": int(args.cost_tol)},
	}

	out_path = Path(args.out)
	if not out_path.is_absolute():
		out_path = base_dir / out_path
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

	# Render a compact table PNG
	table_rows = []
	for r in sorted(rows, key=lambda x: (x["kind"], x["name"])):
		table_rows.append(
			[
				r["name"],
				r["kind"],
				f"{r['match_rate']:.2f}",
				f"{r['confidence']:.3f}",
				f"{r['confidence_drop']:+.3f}",
				f"{r['latency_sec']:.2f}",
				"Y" if r["review_required"] else "N",
			]
		)

	render_table_png(
		headers=["Variant", "Noise", "Match", "Conf", "ΔConf", "Lat(s)", "Review"],
		rows=table_rows,
		title="Robustness: clean vs noisy variants",
		out_path=outputs_dir / "robustness_table.png",
		width=1200,
	)

	print(json.dumps({"out": str(out_path), "table": "outputs/robustness_table.png"}, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
