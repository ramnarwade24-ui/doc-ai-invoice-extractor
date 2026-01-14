from __future__ import annotations

import json
from pathlib import Path

from utils.simple_charts import bar_chart_png, histogram_png


def run_eda(runs_jsonl: Path, outputs_dir: Path) -> None:
	"""Generate judge-friendly EDA artifacts.

	Writes:
	- outputs/eda_outputs/*.png
	- outputs/eda_summary.csv
	"""
	eda_dir = outputs_dir / "eda_outputs"
	eda_dir.mkdir(parents=True, exist_ok=True)
	if not runs_jsonl.exists():
		raise FileNotFoundError(f"No run log found at {runs_jsonl}")

	rows = []
	for line in runs_jsonl.read_text(encoding="utf-8").splitlines():
		if not line.strip():
			continue
		try:
			rows.append(json.loads(line))
		except Exception:
			continue
	if not rows:
		raise ValueError("Run log is empty")

	# Language-wise document distribution
	lang_counts = {}
	for r in rows:
		lang = (r.get("language") or "unknown")
		lang_counts[lang] = int(lang_counts.get(lang, 0) + 1)
	labels = list(lang_counts.keys())
	values = [float(lang_counts[k]) for k in labels]
	bar_chart_png(
		title="Language-wise Document Distribution",
		labels=labels,
		values=values,
		out_path=eda_dir / "language_distribution.png",
	)

	# Latency histogram
	latencies = [float(r.get("processing_time_sec") or 0.0) for r in rows]
	histogram_png(
		title="Latency Distribution (sec)",
		values=latencies,
		bins=20,
		out_path=eda_dir / "latency_histogram.png",
	)

	# OCR confidence distribution
	ocr_confs = [float(r.get("avg_ocr_conf") or 0.0) for r in rows if r.get("avg_ocr_conf") is not None]
	if ocr_confs:
		histogram_png(
			title="OCR Word Confidence Distribution",
			values=ocr_confs,
			bins=20,
			out_path=eda_dir / "ocr_confidence_distribution.png",
		)

	# Region-wise extraction proxy accuracy
	region_hits = {}
	region_total = {}
	for r in rows:
		fields = r.get("fields") or {}
		if not isinstance(fields, dict):
			continue
		for _, meta in fields.items():
			if not isinstance(meta, dict):
				continue
			region = meta.get("region") or "unknown"
			present = meta.get("value") not in (None, "")
			conf = float(meta.get("conf") or 0.0)
			region_total[region] = int(region_total.get(region, 0) + 1)
			if present and conf >= 0.75:
				region_hits[region] = int(region_hits.get(region, 0) + 1)
	if region_total:
		regions = sorted(region_total.keys())
		rates = [float(region_hits.get(k, 0) / max(1, region_total.get(k, 0))) for k in regions]
		bar_chart_png(
			title="Region-wise Extraction High-Confidence Rate (proxy)",
			labels=regions,
			values=rates,
			value_range=(0.0, 1.0),
			out_path=eda_dir / "region_wise_extraction_proxy_accuracy.png",
		)

	# Summary CSV (no pandas dependency)
	def _mean(xs):
		return float(sum(xs) / max(1, len(xs)))

	sorted_lat = sorted(latencies)
	p95 = sorted_lat[int(0.95 * (len(sorted_lat) - 1))] if sorted_lat else 0.0
	avg_conf = _mean([float(r.get("confidence") or 0.0) for r in rows])
	err_rate = _mean([1.0 if bool(r.get("error_flag")) else 0.0 for r in rows])
	avg_cost = _mean([float(r.get("cost_estimate_usd") or 0.0) for r in rows])
	avg_ocr = _mean(ocr_confs) if ocr_confs else 0.0

	csv_path = outputs_dir / "eda_summary.csv"
	csv_path.write_text(
		"docs,avg_confidence,avg_latency_sec,p95_latency_sec,error_rate,avg_cost_usd,avg_ocr_conf\n"
		+ f"{len(rows)},{avg_conf:.6f},{_mean(latencies):.6f},{p95:.6f},{err_rate:.6f},{avg_cost:.6f},{avg_ocr:.6f}\n",
		encoding="utf-8",
	)
