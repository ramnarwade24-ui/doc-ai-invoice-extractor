from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from executable import run_pipeline
from noisy_test import generate_variants
from utils.config import PipelineConfig
from eda import run_eda
from error_analysis import run_error_analysis
from architecture_diagram import generate_architecture_png


st.set_page_config(page_title="DocAI Invoice Extractor", layout="wide")

st.title("Intelligent Document AI — Invoice Extractor")
st.caption("Multilingual (EN/HI/GU) • OCR + Rules + Fuzzy Match + Optional YOLO")

base_dir = Path(__file__).resolve().parent
work_dir = base_dir / "outputs"
work_dir.mkdir(parents=True, exist_ok=True)

tab_extract, tab_robust, tab_judge = st.tabs(["Extract", "Robustness", "Judge Demo"])

with tab_extract:
	uploaded = st.file_uploader("Upload an invoice PDF", type=["pdf"], key="extract_upload")

	col1, col2 = st.columns(2)

	with col1:
		st.subheader("Config")
		dpi = st.slider("DPI", 120, 300, 200, 10, key="extract_dpi")
		max_pages = st.slider("Max pages", 1, 10, 5, 1, key="extract_max_pages")
		yolo_weights = st.text_input("YOLO weights path (optional)", value="", key="extract_yolo")

	run_btn = st.button("Run Extraction", type="primary", disabled=uploaded is None, key="extract_run")

	if uploaded and run_btn:
		pdf_path = work_dir / uploaded.name
		pdf_path.write_bytes(uploaded.getvalue())

		cfg = PipelineConfig(
			dpi=int(dpi),
			max_pages=int(max_pages),
			yolo_weights_path=Path(yolo_weights) if yolo_weights else None,
		)
		result = run_pipeline(pdf_path=str(pdf_path), doc_id=pdf_path.stem, config=cfg)

		# Persist result JSON
		result_path = work_dir / "result.json"
		result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

		with col2:
			st.subheader("Result JSON")
			st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
			st.download_button(
				"Download JSON",
				data=json.dumps(result, ensure_ascii=False, indent=2),
				file_name=f"{pdf_path.stem}_result.json",
				mime="application/json",
			)

		st.subheader("Summary")
		st.write(
			{
				"confidence": result.get("confidence"),
				"review_required": result.get("review_required"),
				"processing_time_sec": result.get("processing_time_sec"),
				"cost_estimate_usd": result.get("cost_estimate_usd"),
			}
		)

		st.subheader("Overlays")
		overlay_dir = work_dir / "overlays"
		matches = sorted(overlay_dir.glob(f"{pdf_path.stem}_page*_overlay.png"))
		if matches:
			for p in matches:
				st.image(str(p), caption=p.name, use_container_width=True)
		else:
			st.info("No overlays generated (set save_overlays=True and ensure OCR ran).")

		st.subheader("EDA")
		try:
			runs_jsonl = work_dir / "runs.jsonl"
			run_eda(runs_jsonl=runs_jsonl, outputs_dir=work_dir)
			edap = work_dir / "eda_outputs"
			for p in sorted(edap.glob("*.png")):
				st.image(str(p), caption=p.name, use_container_width=True)
			csv_path = work_dir / "eda_summary.csv"
			if csv_path.exists():
				st.download_button(
					"Download EDA Summary CSV",
					data=csv_path.read_bytes(),
					file_name="eda_summary.csv",
					mime="text/csv",
				)
		except Exception as e:
			st.warning(f"EDA not available: {e}")

		st.subheader("Error Analysis")
		try:
			err_report = run_error_analysis(runs_jsonl=work_dir / "runs.jsonl", outputs_dir=work_dir)
			st.code(err_report.read_text(encoding="utf-8"), language="json")
			st.download_button(
				"Download Error Report JSON",
				data=err_report.read_bytes(),
				file_name="error_report.json",
				mime="application/json",
			)
			chart = work_dir / "error_distribution.png"
			if chart.exists():
				st.image(str(chart), caption=chart.name, use_container_width=True)
		except Exception as e:
			st.warning(f"Error analysis not available: {e}")

		st.subheader("Architecture Diagram")
		try:
			diag = work_dir / "architecture_diagram.png"
			if not diag.exists():
				generate_architecture_png(diag)
			if diag.exists():
				st.image(str(diag), caption=diag.name, use_container_width=True)
				st.download_button(
					"Download Architecture Diagram",
					data=diag.read_bytes(),
					file_name="architecture_diagram.png",
					mime="image/png",
				)
		except Exception as e:
			st.warning(f"Diagram not available: {e}")

with tab_robust:
	st.subheader("Deterministic robustness harness")
	st.caption("Generates synthetic noisy variants and compares extraction outputs.")

	rob_up = st.file_uploader("Upload a clean invoice PDF", type=["pdf"], key="robust_upload")
	colA, colB = st.columns(2)
	with colA:
		dpi_r = st.slider("DPI", 120, 300, 200, 10, key="robust_dpi")
		seed = st.number_input("Seed", min_value=0, max_value=2_000_000_000, value=1337, step=1)
		profile = st.selectbox("Noise profile", ["mild", "stress"], index=0)
		max_pages_r = st.slider("Max pages (pipeline)", 1, 10, 5, 1, key="robust_max_pages")
	with colB:
		st.markdown("**What this does**")
		st.write(
			[
				"Saves variants under outputs/noisy_tests/<filename>/",
				"Runs the same pipeline on clean + variants",
				"Shows side-by-side images + JSON and flags review_required",
			]
		)

	go = st.button("Generate + Evaluate", type="primary", disabled=rob_up is None, key="robust_go")

	if rob_up and go:
		pdf_path = work_dir / rob_up.name
		pdf_path.write_bytes(rob_up.getvalue())
		out_dir = work_dir / "noisy_tests" / pdf_path.stem

		with st.status("Generating variants...", expanded=False):
			manifest = generate_variants(input_path=pdf_path, out_dir=out_dir, seed=int(seed), dpi=int(dpi_r), profile=str(profile))

		variants = manifest.get("variants") or []
		by_name = {v["name"]: v for v in variants}
		clean_pdf = Path(by_name["clean"]["pdf"]) if "clean" in by_name else None
		if clean_pdf is None:
			st.error("Manifest missing clean variant.")
			st.stop()

		cfg = PipelineConfig(dpi=int(dpi_r), max_pages=int(max_pages_r), deterministic=True, seed=int(seed), run_mode="normal")

		with st.status("Running pipeline on clean...", expanded=False):
			clean_out = run_pipeline(pdf_path=str(clean_pdf), doc_id=f"{pdf_path.stem}_clean", config=cfg)

		st.markdown("**Clean summary**")
		st.write(
			{
				"confidence": clean_out.get("confidence"),
				"review_required": clean_out.get("review_required"),
				"processing_time_sec": clean_out.get("processing_time_sec"),
			}
		)

		variant_names = [v["name"] for v in variants if v["name"] != "clean"]
		pick = st.selectbox("Pick a variant to compare", variant_names, index=0 if variant_names else None)
		if pick:
			v = by_name[pick]
			with st.status(f"Running pipeline on {pick}...", expanded=False):
				noisy_out = run_pipeline(pdf_path=str(v["pdf"]), doc_id=f"{pdf_path.stem}_{pick}", config=cfg)

			img_col1, img_col2 = st.columns(2)
			with img_col1:
				st.markdown("**Clean image**")
				st.image(str(by_name["clean"]["image"]), use_container_width=True)
			with img_col2:
				st.markdown(f"**Variant: {pick}**")
				st.image(str(v["image"]), use_container_width=True)

			res_col1, res_col2 = st.columns(2)
			with res_col1:
				st.markdown("**Clean JSON**")
				st.code(json.dumps(clean_out, ensure_ascii=False, indent=2), language="json")
			with res_col2:
				st.markdown(f"**{pick} JSON**")
				st.code(json.dumps(noisy_out, ensure_ascii=False, indent=2), language="json")

			st.markdown("**Delta**")
			st.write(
				{
					"confidence_clean": clean_out.get("confidence"),
					"confidence_variant": noisy_out.get("confidence"),
					"review_required_clean": clean_out.get("review_required"),
					"review_required_variant": noisy_out.get("review_required"),
				}
			)

		st.info(f"Variants saved in: {out_dir}")


with tab_judge:
	st.subheader("Judge Demo")
	st.caption("Upload a ZIP of PDFs → run deterministic batch extraction → download CSV + summary.")

	# Config picker (default to best_config.json when present)
	best_cfg = base_dir / "best_config.json"
	use_best = bool(best_cfg.exists())
	st.markdown("**Config**")
	col_cfg1, col_cfg2 = st.columns([2, 3])
	with col_cfg1:
		cfg_mode = st.radio(
			"Config source",
			options=["Use best_config.json", "Upload config JSON"],
			index=0 if use_best else 1,
			horizontal=False,
			key="judge_cfg_mode",
		)
	with col_cfg2:
		cfg_upload = None
		if cfg_mode == "Upload config JSON":
			cfg_upload = st.file_uploader("Upload config JSON", type=["json"], key="judge_cfg_upload")
		elif not use_best:
			st.warning("best_config.json not found in doc_ai_project/. Upload a config JSON to run.")

	seed_val = st.number_input("Seed (deterministic)", min_value=0, max_value=2_000_000_000, value=1337, step=1, key="judge_seed")

	zip_up = st.file_uploader("Upload ZIP containing PDFs", type=["zip"], key="judge_zip")
	run_demo = st.button("Run Judge Demo", type="primary", disabled=zip_up is None, key="judge_run")

	def _sha256_bytes(data: bytes) -> str:
		h = hashlib.sha256()
		h.update(data)
		return h.hexdigest()

	def _rows_to_csv(rows: list[dict]) -> str:
		if not rows:
			return ""
		buf = io.StringIO()
		w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
		w.writeheader()
		for r in rows:
			w.writerow(r)
		return buf.getvalue()

	if zip_up and run_demo:
		# Resolve selected config (keep paths relative to doc_ai_project when possible)
		cfg_name = ""
		cfg_sha = ""
		cfg_arg = None
		if cfg_mode == "Use best_config.json":
			if not best_cfg.exists():
				st.error("best_config.json not found. Upload a config JSON.")
				st.stop()
			cfg_name = "best_config.json"
			cfg_bytes = best_cfg.read_bytes()
			cfg_sha = _sha256_bytes(cfg_bytes)
			cfg_arg = "best_config.json"
		else:
			if cfg_upload is None:
				st.error("Please upload a config JSON.")
				st.stop()
			cfg_name = str(cfg_upload.name)
			cfg_bytes = cfg_upload.getvalue()
			cfg_sha = _sha256_bytes(cfg_bytes)
			# Save under doc_ai_project/outputs/ for reproducibility
			cfg_dir = work_dir / "judge_demo"
			cfg_dir.mkdir(parents=True, exist_ok=True)
			saved_cfg = cfg_dir / "selected_config.json"
			saved_cfg.write_bytes(cfg_bytes)
			# Pass relative to doc_ai_project/
			cfg_arg = str(saved_cfg.relative_to(base_dir))

		st.write({"active_config": cfg_name, "config_sha256": cfg_sha})

		# Extract ZIP to a temporary directory (offline, no hardcoded paths)
		tmp_dir = Path(tempfile.mkdtemp(prefix="docai_judge_demo_"))
		try:
			with zipfile.ZipFile(io.BytesIO(zip_up.getvalue())) as z:
				z.extractall(tmp_dir)
		except Exception as e:
			st.error(f"Invalid ZIP: {e}")
			st.stop()

		# Run deterministic pipeline on all PDFs found in the ZIP
		from utils.demo_utils import discover_pdfs, format_table, run_pipeline

		pdfs = discover_pdfs(str(tmp_dir), recursive=True, limit=None, seed=int(seed_val))
		if not pdfs:
			st.error("No PDFs found inside the uploaded ZIP.")
			st.stop()

		with st.status(f"Running pipeline on {len(pdfs)} PDF(s)...", expanded=False):
			results, errors = run_pipeline(pdfs, config_path=cfg_arg, seed=int(seed_val))

		# Display table (requested columns)
		cols = [
			"dealer_name",
			"model_name",
			"horse_power",
			"asset_cost",
			"signature_present",
			"stamp_present",
			"confidence",
			"processing_time_sec",
		]
		view = []
		for r in results:
			view.append({
				"dealer_name": r.get("dealer_name"),
				"model_name": r.get("model_name"),
				"hp": r.get("horse_power"),
				"asset_cost": r.get("asset_cost"),
				"signature": r.get("signature_present"),
				"stamp": r.get("stamp_present"),
				"confidence": r.get("confidence"),
				"latency": r.get("processing_time_sec"),
			})

		st.markdown("**Results**")
		st.dataframe(view, use_container_width=True)

		# Summary metrics + health banner
		latencies = [float(r.get("processing_time_sec") or 0.0) for r in results]
		confs = [float(r.get("confidence") or 0.0) for r in results]
		costs = [float(r.get("cost_estimate_usd") or 0.0) for r in results]
		avg_latency = float(sum(latencies) / max(1, len(latencies)))
		avg_conf = float(sum(confs) / max(1, len(confs)))
		avg_cost = float(sum(costs) / max(1, len(costs)))

		reasons = []
		schema_ok = bool(len(errors) == 0 and len(results) == len(pdfs))
		if not schema_ok:
			reasons.append("schema_validation_failed")
		if not (avg_latency <= 30.0):
			reasons.append("avg_latency_gt_30s")
		if not (avg_cost < 0.01):
			reasons.append("avg_cost_ge_0.01")
		ok = bool(len(reasons) == 0)
		if ok:
			st.success("PASS: schema OK, avg latency ≤ 30s, avg cost < $0.01")
		else:
			st.error("FAIL: " + ", ".join(reasons))

		st.markdown("**Summary**")
		st.write(
			{
				"docs_processed": int(len(results)),
				"errors": int(len(errors)),
				"avg_latency_sec": round(avg_latency, 4),
				"avg_cost_usd": round(avg_cost, 6),
				"avg_confidence": round(avg_conf, 4),
				"status": "PASS" if ok else "FAIL",
				"reasons": reasons,
			}
		)

		if errors:
			st.warning("Some documents failed. See details below.")
			st.code(json.dumps(errors, ensure_ascii=False, indent=2), language="json")

		# Download artifacts
		csv_text = _rows_to_csv(
			[
				{
					"dealer_name": r.get("dealer_name"),
					"model_name": r.get("model_name"),
					"hp": r.get("horse_power"),
					"asset_cost": r.get("asset_cost"),
					"signature": r.get("signature_present"),
					"stamp": r.get("stamp_present"),
					"confidence": r.get("confidence"),
					"latency": r.get("processing_time_sec"),
				}
				for r in results
			]
		)
		summary = {
			"status": "PASS" if ok else "FAIL",
			"reasons": reasons,
			"active_config": cfg_name,
			"config_sha256": cfg_sha,
			"seed": int(seed_val),
			"docs_processed": int(len(results)),
			"errors": int(len(errors)),
			"avg_latency_sec": avg_latency,
			"avg_cost_usd": avg_cost,
			"avg_confidence": avg_conf,
			"counts": {"ok": int(len(results)), "error": int(len(errors))},
		}
		summary_json = json.dumps(summary, ensure_ascii=False, indent=2)

		colA, colB = st.columns(2)
		with colA:
			st.download_button(
				"Download demo_results.csv",
				data=csv_text,
				file_name="demo_results.csv",
				mime="text/csv",
			)
		with colB:
			st.download_button(
				"Download demo_summary.json",
				data=summary_json,
				file_name="demo_summary.json",
				mime="application/json",
			)

		# Optional: if a judge report already exists (from judge_mode), offer it as a download.
		judge_report = (base_dir.parent / "outputs" / "judge_report.json")
		if judge_report.exists():
			st.download_button(
				"Download outputs/judge_report.json",
				data=judge_report.read_bytes(),
				file_name="judge_report.json",
				mime="application/json",
			)

		# Optional: show a terminal-style table too (nice for copy/paste)
		with st.expander("Show CLI-style table"):
			try:
				st.code(format_table(results), language="text")
			except Exception:
				st.info("Table formatting unavailable.")
