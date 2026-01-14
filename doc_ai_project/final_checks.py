from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from utils.resources import load_dealer_master_list, load_model_master_list
from utils.text import normalize_name
from validator import InvoiceOutput


def _range_check(name: str, value: int | None, lo: int, hi: int) -> List[str]:
	if value is None:
		return []
	if value < lo or value > hi:
		return [f"{name} out of range: {value} (expected {lo}..{hi})"]
	return []


def _whitelist_check(name: str, value: str | None, allowed: List[str]) -> List[str]:
	if value is None:
		return []
	allowed_norm = {normalize_name(a) for a in allowed}
	if normalize_name(value) not in allowed_norm:
		return [f"{name} not in whitelist: {value}"]
	return []


def run_checks(*, json_path: Path, base_dir: Path, strict: bool) -> Dict[str, Any]:
	obj = json.loads(json_path.read_text(encoding="utf-8"))

	issues: List[str] = []
	try:
		out = InvoiceOutput.model_validate(obj)
	except ValidationError as e:
		return {"ok": False, "schema_ok": False, "issues": ["schema_validation_failed"], "details": json.loads(e.json())}

	# Numeric sanity / range checks
	issues.extend(_range_check("horse_power", out.fields.horse_power, 1, 300))
	issues.extend(_range_check("asset_cost", out.fields.asset_cost, 1000, 2_000_000_000))

	# Simple sanity: if cost present, it should be >= 0
	if out.cost_estimate_usd < 0:
		issues.append("cost_estimate_usd negative")
	if out.processing_time_sec < 0:
		issues.append("processing_time_sec negative")

	# Whitelist validations (warn by default; strict can fail)
	dealers = load_dealer_master_list(base_dir)
	models = load_model_master_list(base_dir)
	issues.extend(_whitelist_check("dealer_name", out.fields.dealer_name, dealers))
	issues.extend(_whitelist_check("model_name", out.fields.model_name, models))

	ok = len(issues) == 0 if strict else True
	return {
		"ok": bool(ok),
		"schema_ok": True,
		"strict": bool(strict),
		"review_required": bool(out.review_required),
		"confidence": float(out.confidence),
		"issues": issues,
	}


def main() -> int:
	p = argparse.ArgumentParser(description="Final sanity validation for DocAI outputs")
	p.add_argument("--json", required=True, help="Path to output JSON from executable")
	p.add_argument("--strict", action="store_true", help="Fail (exit 2) on any issue, including whitelist")
	p.add_argument("--out", default="outputs/final_checks_report.json", help="Write report JSON")
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	json_path = Path(args.json)
	if not json_path.is_absolute():
		json_path = base_dir / json_path

	out_path = Path(args.out)
	if not out_path.is_absolute():
		out_path = base_dir / out_path
	out_path.parent.mkdir(parents=True, exist_ok=True)

	rep = run_checks(json_path=json_path, base_dir=base_dir, strict=bool(args.strict))
	out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

	print(json.dumps(rep, ensure_ascii=False, indent=2))
	return 0 if rep.get("ok") else 2


if __name__ == "__main__":
	raise SystemExit(main())
