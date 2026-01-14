from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageDraw, ImageFont

from utils.simple_charts import ChartStyle, bar_chart_png


def render_scorecard_png(*, summary: Dict[str, Any], out_path: Path) -> Path:
	"""Create a single PNG with headline metrics + per-field error rates."""
	out_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		font = ImageFont.load_default()
	except Exception:
		font = None

	# Generate bar chart for field error rates
	errs = summary.get("field_error_rate") or {}
	chart_path = out_path.parent / "_scorecard_field_errors.png"
	bar_chart_png(
		labels=list(errs.keys()),
		values=[float(v) for v in errs.values()],
		title="Field Error Rate (lower is better)",
		out_path=chart_path,
		style=ChartStyle(width=1100, height=420),
	)
	chart = Image.open(chart_path).convert("RGB")

	# Header panel
	header_h = 180
	img = Image.new("RGB", (chart.width, header_h + chart.height), (255, 255, 255))
	d = ImageDraw.Draw(img)

	title = "Hackathon Scorecard"
	d.text((14, 12), title, fill=(20, 20, 20), font=font)

	dla = float(summary.get("dla") or 0.0)
	lat_avg = float(summary.get("latency_avg_sec") or 0.0)
	lat_p95 = float(summary.get("latency_p95_sec") or 0.0)
	cost = float(summary.get("cost_avg_usd") or 0.0)
	final_score = float(summary.get("final_score") or 0.0)
	comps = summary.get("final_score_components") or {}

	lines = [
		f"DLA: {dla:.3f}",
		f"Latency avg/p95 (sec): {lat_avg:.3f} / {lat_p95:.3f}",
		f"Cost avg (USD): {cost:.5f}",
		f"Final score: {final_score:.3f}",
		f"Weights (DLA/lat/cost): {comps.get('w_dla', 0):.2f} / {comps.get('w_latency', 0):.2f} / {comps.get('w_cost', 0):.2f}",
	]

	y = 48
	for line in lines:
		d.text((14, y), line, fill=(40, 40, 40), font=font)
		y += 24

	# Paste chart
	img.paste(chart, (0, header_h))
	img.save(out_path)
	try:
		chart_path.unlink(missing_ok=True)
	except Exception:
		pass
	return out_path
