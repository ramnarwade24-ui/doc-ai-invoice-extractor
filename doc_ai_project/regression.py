from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from utils.config import PipelineConfig


def main() -> int:
	p = argparse.ArgumentParser(description="Regression gate on noisy-document robustness")
	p.add_argument("--report", default="outputs/robustness_report.json", help="robustness_eval.py output")
	p.add_argument("--require-no-new-review", action="store_true", help="Fail if a noisy variant becomes review_required when clean wasn't")
	p.add_argument("--max-drop", type=float, default=0.0, help="Optional allowed confidence drop vs clean (0 means ignore)")
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	report_path = Path(args.report)
	if not report_path.is_absolute():
		report_path = base_dir / report_path

	obj = json.loads(report_path.read_text(encoding="utf-8"))
	clean = obj.get("clean") or {}
	rows = obj.get("rows") or []

	clean_review = bool(clean.get("review_required"))
	clean_conf = float(clean.get("confidence") or 0.0)

	issues: List[str] = []
	failed_variants: List[str] = []

	for r in rows:
		name = r.get("name")
		if name == "clean":
			continue
		matches = r.get("matches") or {}
		bad_fields = [k for k, ok in matches.items() if ok is False]
		if bad_fields:
			issues.append(f"field_mismatch:{name}:{bad_fields}")
			failed_variants.append(str(name))

		noisy_conf = float(r.get("confidence") or 0.0)
		noisy_review = bool(r.get("review_required"))
		if args.require_no_new_review and (not clean_review) and noisy_review:
			issues.append(f"new_review_required:{name}")
			failed_variants.append(str(name))

		if float(args.max_drop) > 0.0:
			if (clean_conf - noisy_conf) > float(args.max_drop):
				issues.append(f"confidence_drop_exceeds:{name}:{clean_conf - noisy_conf:.3f}>")
				failed_variants.append(str(name))

	rep = {
		"ok": len(issues) == 0,
		"report": str(report_path),
		"issues": issues,
		"failed_variants": sorted(set(failed_variants)),
	}
	out_path = base_dir / "outputs" / "regression_report.json"
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

	print(json.dumps(rep, ensure_ascii=False, indent=2))
	return 0 if rep["ok"] else 2


if __name__ == "__main__":
	raise SystemExit(main())
