from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from layout import Line, StructuredLayout
from utils.resources import load_dealer_master_list, load_model_master_list
from utils.text import (
	digits_only,
	normalize_keywords,
	normalize_name,
	normalize_spaces,
	parse_hp,
	strip_currency_tokens,
)


MONEY_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{2,3})+|\d+)(?!\d)")


_ADAPTER_CACHE: Dict[str, object] = {}


def _load_dataset_adapters(base_dir: Path) -> Dict[str, object]:
	"""Load dataset adapters JSON if present.

	Expected location: repo-root relative doc_ai_project/outputs/dataset_adapters.json
	"""
	key = "dataset_adapters"
	if key in _ADAPTER_CACHE:
		return _ADAPTER_CACHE[key]  # type: ignore[return-value]
	try:
		repo_root = base_dir.parent
		path = repo_root / "doc_ai_project" / "outputs" / "dataset_adapters.json"
		if not path.exists():
			_ADAPTER_CACHE[key] = {}
			return {}
		import json

		obj = json.loads(path.read_text(encoding="utf-8"))
		if not isinstance(obj, dict):
			_ADAPTER_CACHE[key] = {}
			return {}
		_ADAPTER_CACHE[key] = obj
		return obj  # type: ignore[return-value]
	except Exception:
		_ADAPTER_CACHE[key] = {}
		return {}


def _apply_replacements(text: str, replacements: List[Tuple[str, str]]) -> str:
	out = text or ""
	for a, b in replacements:
		out = out.replace(a, b)
	return out


def _adapt_text(*, base_dir: Path, kind: str, text: str) -> str:
	ad = _load_dataset_adapters(base_dir)
	conf = ad.get(kind) if isinstance(ad, dict) else None
	if not isinstance(conf, dict):
		return text
	reps = conf.get("replacements")
	if isinstance(reps, list):
		pairs: List[Tuple[str, str]] = []
		for it in reps:
			if isinstance(it, list) and len(it) == 2:
				pairs.append((str(it[0]), str(it[1])))
			elif isinstance(it, tuple) and len(it) == 2:
				pairs.append((str(it[0]), str(it[1])))
			elif isinstance(it, dict) and "from" in it and "to" in it:
				pairs.append((str(it["from"]), str(it["to"])))
			elif isinstance(it, (str, int)):
				# ignore malformed
				continue
			else:
				continue
			
		return _apply_replacements(text, pairs)
	return text


MONEY_RELAXED_RE = re.compile(r"(?<!\d)(?:\d[\d,\.\s]{2,}\d|\d+)(?!\d)")


def _find_money_numbers(text: str) -> List[str]:
	"""More tolerant money detection than MONEY_RE for real-world datasets.

	We still parse via digits_only() so separators are safe.
	"""
	return MONEY_RELAXED_RE.findall(text or "")


@dataclass(frozen=True)
class FieldCandidate:
	value: object
	conf: float
	ocr_conf: float
	rule_conf: float
	region_conf: float
	region: str
	used_fallback: bool
	source: str
	bbox: Optional[Tuple[int, int, int, int]] = None

	def as_log_dict(self) -> Dict[str, object]:
		return {
			"value": self.value,
			"conf": float(self.conf),
			"ocr_conf": float(self.ocr_conf),
			"rule_conf": float(self.rule_conf),
			"region_conf": float(self.region_conf),
			"region": self.region,
			"used_fallback": bool(self.used_fallback),
			"source": self.source,
			"bbox": list(self.bbox) if self.bbox else [],
		}


def _best_fuzzy_match(text: str, choices: List[str]) -> tuple[Optional[str], float]:
	if not text or not choices:
		return None, 0.0
	match = process.extractOne(
		query=text,
		choices=choices,
		scorer=fuzz.token_set_ratio,
		processor=normalize_name,
	)
	if not match:
		return None, 0.0
	choice, score, _ = match
	return str(choice), float(score)


def _region_weight(
	region: str,
	expected: str,
	overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
	if overrides and expected in overrides:
		m = overrides.get(expected) or {}
		if region in m:
			return float(m[region])
		# allow a catch-all
		if "*" in m:
			return float(m["*"])
	# default heuristic weights
	if region == expected:
		return 1.0
	if expected == "table" and region in ("body",):
		return 0.85
	if expected == "footer" and region in ("table", "body"):
		return 0.75
	if expected == "header" and region in ("body",):
		return 0.7
	return 0.6


def extract_dealer_name(
	layout: StructuredLayout,
	base_dir: Path,
	threshold: int = 90,
	region_weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> FieldCandidate:
	masters = load_dealer_master_list(base_dir)
	# Region priority: header -> body -> footer
	search = [("header", layout.regions["header"].lines), ("body", layout.regions["body"].lines), ("footer", layout.regions["footer"].lines)]
	# Prefer dealer identity from the first page (common in multipage invoices)
	first_page = 0
	try:
		first_page = min([ln.page_index for ln in layout.all_lines]) if layout.all_lines else 0
	except Exception:
		first_page = 0

	label_re = re.compile(r"(?i)\b(dealer|seller|vendor|supplier)\b|विक्रेता|डीलर|विक्रय|ડિલર|વિક્રેતા")

	best_val: Optional[str] = None
	best_score = 0.0
	best_ln: Optional[Line] = None
	best_region = ""
	used_fallback = False

	for region, lines in search:
		# First-page preference
		lines_pref = [ln for ln in lines if ln.page_index == first_page] or list(lines)
		# First pass: label-based extraction
		for ln in lines_pref[:60]:
			if not label_re.search(ln.text):
				continue
			parts = re.split(r"[:\-]", ln.text, maxsplit=1)
			candidate = normalize_spaces(parts[1] if len(parts) == 2 else ln.text)
			candidate = _adapt_text(base_dir=base_dir, kind="dealer", text=candidate)
			m, s = _best_fuzzy_match(candidate, masters)
			if m and s > best_score:
				best_val, best_score, best_ln, best_region = m, s, ln, region
				used_fallback = False

		# Second pass: general fuzzy over region
		if best_score < threshold:
			for ln in lines_pref[:80]:
				m, s = _best_fuzzy_match(_adapt_text(base_dir=base_dir, kind="dealer", text=ln.text), masters)
				if m and s > best_score:
					best_val, best_score, best_ln, best_region = m, s, ln, region
					used_fallback = True

		if best_score >= threshold:
			break

	ocr_conf = float(best_ln.avg_conf) if best_ln else 0.0
	rule_conf = min(1.0, best_score / 100.0)
	region_conf = _region_weight(best_region or "body", expected="header", overrides=region_weight_overrides) * float(
		layout.regions[best_region or "body"].confidence if (best_region or "body") in layout.regions else 0.6
	)
	conf = float(round(ocr_conf * rule_conf * region_conf, 4))

	if best_val is None or best_score < threshold:
		return FieldCandidate(
			value=None,
			conf=conf * 0.5,
			ocr_conf=ocr_conf,
			rule_conf=rule_conf,
			region_conf=region_conf,
			region=best_region or "unknown",
			used_fallback=True,
			source=best_ln.text if best_ln else "",
			bbox=best_ln.bbox if best_ln else None,
		)

	return FieldCandidate(
		value=best_val,
		conf=conf,
		ocr_conf=ocr_conf,
		rule_conf=rule_conf,
		region_conf=region_conf,
		region=best_region,
		used_fallback=used_fallback,
		source=best_ln.text if best_ln else "",
		bbox=best_ln.bbox if best_ln else None,
	)


def extract_model_name(
	layout: StructuredLayout,
	base_dir: Path,
	region_weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> FieldCandidate:
	models = load_model_master_list(base_dir)
	priorities = ["table", "body", "header", "footer"]

	for region in priorities:
		for ln in layout.regions[region].lines:
			line_text = _adapt_text(base_dir=base_dir, kind="model", text=ln.text)
			for m in models:
				if normalize_spaces(m).casefold() in line_text.casefold():
					ocr_conf = float(ln.avg_conf)
					rule_conf = 0.98
					region_conf = _region_weight(region, expected="table", overrides=region_weight_overrides) * float(layout.regions[region].confidence)
					conf = float(round(ocr_conf * rule_conf * region_conf, 4))
					return FieldCandidate(
						value=m,
						conf=conf,
						ocr_conf=ocr_conf,
						rule_conf=rule_conf,
						region_conf=region_conf,
						region=region,
						used_fallback=False,
						source=ln.text,
						bbox=ln.bbox,
					)

	# Label-based fallback
	label_re = re.compile(r"(?i)\bmodel\b|मॉडल|माडल|मोडल|મોડલ|મોડેલ")
	for region in priorities:
		for ln in layout.regions[region].lines:
			txt = normalize_keywords(_adapt_text(base_dir=base_dir, kind="model", text=ln.text))
			if not label_re.search(txt):
				continue
			parts = re.split(r"[:\-]", ln.text, maxsplit=1)
			candidate = normalize_spaces(parts[1] if len(parts) == 2 else "")
			candidate = _adapt_text(base_dir=base_dir, kind="model", text=candidate)
			if candidate:
				ocr_conf = float(ln.avg_conf)
				rule_conf = 0.55
				region_conf = _region_weight(region, expected="table", overrides=region_weight_overrides) * float(layout.regions[region].confidence)
				conf = float(round(ocr_conf * rule_conf * region_conf, 4))
				return FieldCandidate(
					value=candidate,
					conf=conf,
					ocr_conf=ocr_conf,
					rule_conf=rule_conf,
					region_conf=region_conf,
					region=region,
					used_fallback=True,
					source=ln.text,
					bbox=ln.bbox,
				)

	return FieldCandidate(value=None, conf=0.0, ocr_conf=0.0, rule_conf=0.0, region_conf=0.0, region="unknown", used_fallback=True, source="")


def extract_horse_power(
	layout: StructuredLayout,
	region_weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> FieldCandidate:
	# Search across body+table first
	regions = ["table", "body", "header", "footer"]
	best: Optional[Tuple[int, Line, str]] = None
	best_score = 0.0

	for region in regions:
		for ln in layout.regions[region].lines:
			hp = parse_hp(ln.text)
			if hp is None:
				continue
			score = float(ln.avg_conf) * _region_weight(region, expected="table", overrides=region_weight_overrides)
			if score > best_score:
				best = (hp, ln, region)
				best_score = score

	if not best:
		return FieldCandidate(value=None, conf=0.0, ocr_conf=0.0, rule_conf=0.0, region_conf=0.0, region="unknown", used_fallback=True, source="")

	hp, ln, region = best
	ocr_conf = float(ln.avg_conf)
	rule_conf = 0.9
	region_conf = _region_weight(region, expected="table", overrides=region_weight_overrides) * float(layout.regions[region].confidence)
	conf = float(round(ocr_conf * rule_conf * region_conf, 4))
	return FieldCandidate(
		value=hp,
		conf=conf,
		ocr_conf=ocr_conf,
		rule_conf=rule_conf,
		region_conf=region_conf,
		region=region,
		used_fallback=False,
		source=ln.text,
		bbox=ln.bbox,
	)


def extract_asset_cost(
	layout: StructuredLayout,
	region_weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> FieldCandidate:
	base_dir = Path(__file__).resolve().parent
	# Prefer footer totals
	label_re = re.compile(r"(?i)asset\s*cost|invoice\s*amount|grand\s*total|net\s*amount|total\b|amount\b|राशि|कुल|कुल\s*राशि|કુલ|રકમ")
	regions = ["footer", "table", "body", "header"]
	last_page = 0
	try:
		last_page = max([ln.page_index for ln in layout.all_lines]) if layout.all_lines else 0
	except Exception:
		last_page = 0

	cands: List[FieldCandidate] = []
	for region in regions:
		for ln in layout.regions[region].lines:
			# normalize vernacular keywords (helps label matching)
			norm_line = normalize_keywords(_adapt_text(base_dir=base_dir, kind="price", text=ln.text))
			txt = strip_currency_tokens(norm_line)
			nums = _find_money_numbers(txt)
			if not nums:
				continue
			vals = [int(digits_only(n) or "0") for n in nums]
			vals = [v for v in vals if v > 0]
			if not vals:
				continue
			val = max(vals)
			ocr_conf = float(ln.avg_conf)
			rule_conf = 0.65
			is_totalish = bool(label_re.search(norm_line))
			if is_totalish:
				rule_conf = 0.92
			# Multi-page intelligence: totals usually appear on the last page
			if is_totalish and ln.page_index != last_page and last_page > 0:
				rule_conf *= 0.55
			# Prefer last-page candidates
			page_boost = 1.15 if ln.page_index == last_page else 0.90
			region_conf = _region_weight(region, expected="footer", overrides=region_weight_overrides) * float(layout.regions[region].confidence)
			conf = float(round(ocr_conf * rule_conf * region_conf * page_boost, 4))
			cands.append(
				FieldCandidate(
					value=val,
					conf=conf,
					ocr_conf=ocr_conf,
					rule_conf=rule_conf,
					region_conf=region_conf,
					region=region,
					used_fallback=not bool(is_totalish),
					source=ln.text,
					bbox=ln.bbox,
				)
			)

	if not cands:
		return FieldCandidate(value=None, conf=0.0, ocr_conf=0.0, rule_conf=0.0, region_conf=0.0, region="unknown", used_fallback=True, source="")

	# Prefer high conf; tie-break by larger value
	cands.sort(key=lambda c: (c.conf, int(c.value or 0)), reverse=True)
	return cands[0]


def aggregate_confidence(per_field: Dict[str, FieldCandidate]) -> float:
	# Weighted average with missing-field penalty.
	weights = {
		"dealer_name": 0.25,
		"model_name": 0.25,
		"horse_power": 0.2,
		"asset_cost": 0.3,
	}
	acc = 0.0
	ws = 0.0
	missing_penalty = 0.0
	for k, w in weights.items():
		fc = per_field.get(k)
		if not fc or fc.value is None:
			missing_penalty += 0.08
			continue
		acc += w * float(fc.conf)
		ws += w

	base = acc / ws if ws > 0 else 0.0
	out = max(0.0, min(1.0, base - missing_penalty))
	return float(round(out, 4))


def calibrate_doc_confidence(
	*,
	per_field: Dict[str, FieldCandidate],
	layout: StructuredLayout,
	base_conf: float,
	ocr_fallback_used: bool,
	run_mode: str,
) -> float:
	"""Apply hackathon-style confidence calibration.

	- Penalize missing regions
	- Penalize OCR fallback
	- Penalize replay mode
	- Boost consistent multi-signal fields
	"""
	conf = float(base_conf)

	# Missing region penalties
	missing_regions = 0
	for rn in ("header", "body", "table", "footer"):
		reg = layout.regions.get(rn)
		if not reg or not reg.lines:
			missing_regions += 1
	conf -= 0.03 * float(missing_regions)

	# OCR fallback penalty (scanned PDFs without PaddleOCR, etc.)
	if ocr_fallback_used:
		conf -= 0.06

	# Replay mode penalty (to avoid overconfident recovery runs)
	if str(run_mode).strip().lower() == "replay":
		conf -= 0.03

	# Boost if multiple fields are high-confidence across signals
	bonus = 0.0
	for fc in per_field.values():
		if fc.value is None:
			continue
		if float(fc.ocr_conf) >= 0.85 and float(fc.rule_conf) >= 0.85 and float(fc.region_conf) >= 0.85:
			bonus += 0.02
	conf += min(0.06, bonus)

	conf = max(0.0, min(1.0, conf))
	return float(round(conf, 4))
