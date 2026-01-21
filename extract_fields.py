"""Lightweight regex-based invoice field extraction.

This module is intentionally dependency-free and operates on full OCR text.

Expected output schema:
{
  "dealer_name": str | None,
  "model_name": str | None,
  "horse_power": int | None,
  "asset_cost": int | None
}
"""

from __future__ import annotations

import re
from typing import Any, Dict


_DEALER_RE = re.compile(r"(?i)\b(?:ltd|corporation|tractors)\b\.?\s*")
_MODEL_RE = re.compile(r"(?i)\btractor\b")
_HP_RE = re.compile(r"(?i)(\d+)\s*HP")
# Matches either comma-grouped numbers (incl. Indian grouping) or plain digits.
_MONEY_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{2,3})+|\d+)(?!\d)")


def _digits_only(s: str) -> str:
	return re.sub(r"\D+", "", s or "")


def extract_fields(ocr_text: str) -> Dict[str, Any]:
	"""Extract invoice fields from full OCR text using simple regex rules.

	Rules:
	- dealer_name: first line containing keywords ["Ltd", "Corporation", "Tractors"]
	- model_name: line containing "Tractor"
	- horse_power: regex "(\\d+)\\s*HP"
	- asset_cost: largest number found in format like 8,01,815 or 801815
	"""
	text = ocr_text or ""
	lines = [ln.strip() for ln in text.splitlines()]
	lines = [ln for ln in lines if ln]

	dealer_name: str | None = None
	model_name: str | None = None
	horse_power: int | None = None
	asset_cost: int | None = None

	for ln in lines:
		if dealer_name is None and _DEALER_RE.search(ln):
			dealer_name = ln
		if model_name is None and _MODEL_RE.search(ln):
			model_name = ln
		if horse_power is None:
			m = _HP_RE.search(ln)
			if m:
				try:
					horse_power = int(m.group(1))
				except Exception:
					horse_power = None
		if dealer_name is not None and model_name is not None and horse_power is not None:
			break

	vals = []
	for raw in _MONEY_RE.findall(text):
		try:
			v = int(_digits_only(raw) or "0")
		except Exception:
			continue
		if v > 0:
			vals.append(v)
	if vals:
		asset_cost = max(vals)

	return {
		"dealer_name": dealer_name,
		"model_name": model_name,
		"horse_power": horse_power,
		"asset_cost": asset_cost,
	}
