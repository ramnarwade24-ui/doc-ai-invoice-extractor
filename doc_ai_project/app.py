from __future__ import annotations

import json
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

tab_extract, tab_robust = st.tabs(["Extract", "Robustness"])

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
