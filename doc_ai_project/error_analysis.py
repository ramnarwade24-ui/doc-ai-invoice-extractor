from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.simple_charts import bar_chart_png


@dataclass(frozen=True)
class ErrorTag:
	category: str  # ocr/layout/extraction/vision
	detail: str


def tag_failure_case(fields: Dict[str, object], confidence: float) -> List[ErrorTag]:
	out: List[ErrorTag] = []
	missing = [k for k, v in fields.items() if v in (None, "")]
	if missing:
		out.append(ErrorTag(category="extraction", detail=f"missing:{','.join(missing)}"))
	if confidence < 0.75:
		out.append(ErrorTag(category="extraction", detail="low_confidence"))
	return out


def _classify_failure(row: Dict[str, Any]) -> Tuple[str, List[str]]:
	"""Best-effort failure categorization for runs without ground truth."""
	reasons: List[str] = []
	# Validation (latency/cost)
	if row.get("latency_ok") is False:
		reasons.append("latency_exceeded")
		return "Validation", reasons
	if row.get("cost_ok") is False:
		reasons.append("cost_exceeded")
		return "Validation", reasons

	# OCR
	if int(row.get("ocr_failures") or 0) > 0:
		reasons.append("ocr_failures")
		return "OCR", reasons
	if float(row.get("avg_ocr_conf") or 0.0) < 0.55:
		reasons.append("low_ocr_conf")
		return "OCR", reasons

	# Layout
	region_counts = row.get("region_line_counts") or {}
	if isinstance(region_counts, dict) and sum(int(v or 0) for v in region_counts.values()) == 0:
		reasons.append("no_region_lines")
		return "Layout", reasons

	# Vision
	if bool(row.get("yolo_used")) and not (row.get("signature_present") or row.get("stamp_present")):
		reasons.append("no_detections")
		return "Vision", reasons

	# Extraction
	fields = row.get("fields") or {}
	missing = []
	if isinstance(fields, dict):
		for fname, meta in fields.items():
			val = (meta or {}).get("value") if isinstance(meta, dict) else None
			if val in (None, ""):
				missing.append(fname)
	if missing:
		reasons.append("missing_fields:" + ",".join(missing))
		return "Extraction", reasons

	# Default: OK
	return "OK", reasons


def run_error_analysis(runs_jsonl: Path, outputs_dir: Path) -> Path:
	outputs_dir.mkdir(parents=True, exist_ok=True)
	if not runs_jsonl.exists():
		raise FileNotFoundError(f"No run log found at {runs_jsonl}")

	rows: List[Dict[str, Any]] = []
	for line in runs_jsonl.read_text(encoding="utf-8").splitlines():
		if not line.strip():
			continue
		try:
			rows.append(json.loads(line))
		except Exception:
			continue
	if not rows:
		raise ValueError("Run log is empty")

	# Categorize failures
	cat_counts: Dict[str, int] = {}
	for r in rows:
		cat, _reasons = _classify_failure(r)
		cat_counts[cat] = int(cat_counts.get(cat, 0) + 1)

	# Field-wise proxy accuracy report: present + conf>=0.75
	field_present: Dict[str, int] = {}
	field_high_conf: Dict[str, int] = {}
	field_conf_sum: Dict[str, float] = {}
	field_total: Dict[str, int] = {}
	for r in rows:
		fields = r.get("fields") or {}
		if not isinstance(fields, dict):
			continue
		for fname, meta in fields.items():
			if not isinstance(meta, dict):
				continue
			present = meta.get("value") not in (None, "")
			conf = float(meta.get("conf") or 0.0)
			field_total[fname] = int(field_total.get(fname, 0) + 1)
			field_conf_sum[fname] = float(field_conf_sum.get(fname, 0.0) + conf)
			if present:
				field_present[fname] = int(field_present.get(fname, 0) + 1)
				if conf >= 0.75:
					field_high_conf[fname] = int(field_high_conf.get(fname, 0) + 1)

	field_report: Dict[str, Dict[str, float]] = {}
	for fname in sorted(field_total.keys()):
		tot = max(1, field_total.get(fname, 0))
		field_report[fname] = {
			"present_rate": float(field_present.get(fname, 0) / tot),
			"high_conf_rate": float(field_high_conf.get(fname, 0) / tot),
			"avg_conf": float(field_conf_sum.get(fname, 0.0) / tot),
		}

	# Error distribution chart
	order = ["OK", "Extraction", "OCR", "Layout", "Vision", "Validation"]
	labels = [c for c in order if c in cat_counts]
	values = [float(cat_counts.get(c, 0)) for c in labels]
	bar_chart_png(
		title="Failure Category Distribution",
		labels=labels,
		values=values,
		out_path=outputs_dir / "error_distribution.png",
	)

	# JSON report
	report = {
		"total_docs": int(len(rows)),
		"failure_category_counts": cat_counts,
		"field_wise_proxy_accuracy": field_report,
		"notes": "Field-wise metrics are proxy (present + conf>=0.75) without ground truth.",
	}
	json_path = outputs_dir / "error_report.json"
	json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

	return json_path
