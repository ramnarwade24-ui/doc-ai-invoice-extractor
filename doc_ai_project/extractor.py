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
	keyword_anchored: bool = False,
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

		# Second pass: general fuzzy over region (skip in keyword-anchored mode)
		if (not keyword_anchored) and best_score < threshold:
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
		# In noisy layouts, prefer returning a keyword-anchored best-effort candidate
		# rather than None (still low confidence so downstream can mark review_required).
		if keyword_anchored and best_val is not None and best_ln is not None:
			return FieldCandidate(
				value=best_val,
				conf=float(round(conf * 0.75, 4)),
				ocr_conf=ocr_conf,
				rule_conf=rule_conf,
				region_conf=region_conf,
				region=best_region or "unknown",
				used_fallback=True,
				source=best_ln.text,
				bbox=best_ln.bbox,
			)
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
	keyword_anchored: bool = False,
) -> FieldCandidate:
	models = load_model_master_list(base_dir)
	priorities = ["table", "body", "header", "footer"]

	# In noisy layouts, try keyword-anchored label extraction first.
	if keyword_anchored:
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
					rule_conf = 0.6
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
	header_first: bool = False,
) -> FieldCandidate:
	base_dir = Path(__file__).resolve().parent
	# Prefer footer totals
	label_re = re.compile(r"(?i)asset\s*cost|invoice\s*amount|grand\s*total|net\s*amount|total\b|amount\b|राशि|कुल|कुल\s*राशि|કુલ|રકમ")
	regions = ["header", "footer", "table", "body"] if header_first else ["footer", "table", "body", "header"]
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


def extract_fields(ocr_text: str) -> Dict[str, object]:
	r"""Extract invoice fields from full OCR text using simple regex rules.

	Input: full OCR text string
	Return:
	{
		"dealer_name": str | None,
		"model_name": str | None,
		"horse_power": int | None,
		"asset_cost": int | None,
	}

	Rules:
	- dealer_name: first line containing keywords ["Ltd", "Corporation", "Tractors"]
	- model_name: line containing "Tractor"
	- horse_power: regex r"(\d+)\s*HP"
	- asset_cost: largest number found in format like 8,01,815 or 801815
	"""
	text = ocr_text or ""
	# Translate common Indic digits to ASCII so regex + parse_hp work.
	_digit_map = str.maketrans(
		{
			"०": "0",
			"१": "1",
			"२": "2",
			"३": "3",
			"४": "4",
			"५": "5",
			"६": "6",
			"७": "7",
			"८": "8",
			"९": "9",
			"૦": "0",
			"૧": "1",
			"૨": "2",
			"૩": "3",
			"૪": "4",
			"૫": "5",
			"૬": "6",
			"૭": "7",
			"૮": "8",
			"૯": "9",
		}
	)
	text = text.translate(_digit_map)
	# Normalize a very common OCR confusion: "yo HP" meaning "40 HP".
	text = re.sub(r"(?i)\byo\b(?=\s*(?:hp|h\.\s*p\.?)\b)", "40", text)
	# Normalize OCR corruption near HP, e.g. "4?+ 0 HP" or "4 0 HP" -> "40 HP".
	text = re.sub(
		r"(?i)\b(\d)\s*[?+]+\s*(\d)(?=\s*(?:hp|h\.\s*p\.?)\b)",
		r"\1\2",
		text,
	)
	text = re.sub(
		r"(?i)\b(\d)\s+(\d)(?=\s*(?:hp|h\.\s*p\.?)\b)",
		r"\1\2",
		text,
	)
	lines = [normalize_spaces(ln).strip() for ln in text.splitlines()]
	lines = [ln for ln in lines if ln]
	# Tesseract TSV/layout grouping sometimes splits the HP value across lines,
	# e.g. "... 4?+" on one line and "0 HP" on the next. Fuse such pairs.
	fused_lines: List[str] = []
	i = 0
	while i < len(lines):
		if i + 1 < len(lines):
			prev = lines[i]
			nxt = lines[i + 1]
			if re.search(r"(?i)\b\d\s*[?+]*\s*$", prev) and re.search(r"(?i)^\d\s*(?:hp|h\.\s*p\.?)\b", nxt):
				fused_lines.append(f"{prev} {nxt}")
				i += 2
				continue
		fused_lines.append(lines[i])
		i += 1
	lines = fused_lines

	dealer_name: str | None = None
	model_name: str | None = None
	horse_power: int | None = None
	asset_cost: int | None = None

	dealer_re = re.compile(r"(?i)\b(?:ltd|corporation|tractors)\b")
	model_re = re.compile(r"(?i)\btractor\b")
	# HP patterns seen in OCR: "39 HP", "HP- 39", "HP.: 39", "H.P. 39", etc.
	hp_re = re.compile(r"(?i)(\d{1,4})\s*(?:hp|h\.\s*p\.?)\b")
	hp_rev_re = re.compile(r"(?i)\b(?:hp|h\.\s*p\.?)\s*[\-:./]*\s*(\d{1,4})\b")
	# Amount tokens: support decimals like 550,000.00
	money_re = re.compile(r"(?i)(?<!\d)(?:\d[\d,]*\.\d{2}|\d{1,3}(?:,\d{2,3})+|\d+)(?!\d)")

	def _clean_line(s: str) -> str:
		s = normalize_spaces(s).strip()
		s = re.sub(r"(?i)^the\s+", "", s)
		# Strip obvious OCR separators at the ends.
		s = re.sub(r"[\s\|,:;]+$", "", s)
		return s.strip()

	def _normalize_dealer_line(s: str) -> str:
		s = _clean_line(s)
		m = re.search(r"(?i)\bauthori[sz]ed\s+dealer\s+for\s+(.+)$", s)
		if m:
			return _clean_line(m.group(1))
		m = re.search(r"(?i)\bdealer\s+for\s+(.+)$", s)
		if m:
			return _clean_line(m.group(1))
		return s

	def _normalize_hp(raw: str) -> int | None:
		d = digits_only(raw)
		if not d:
			return None
		try:
			val = int(d)
		except Exception:
			return None
		# Typical tractor HP range; OCR sometimes inserts an extra digit.
		if 10 <= val <= 125:
			return val
		if val > 125 and len(d) >= 3:
			cands: List[int] = []
			for i in range(len(d)):
				cand_s = d[:i] + d[i + 1 :]
				if not cand_s:
					continue
				try:
					cand = int(cand_s)
				except Exception:
					continue
				if 10 <= cand <= 125:
					cands.append(cand)
			if cands:
				from collections import Counter

				ctr = Counter(cands)
				best_count = max(ctr.values())
				best = [v for v, c in ctr.items() if c == best_count]
				return max(best)
		# Outside plausible range: treat as noise.
		return val if 10 <= val <= 125 else None

	def _parse_money_token(raw: str) -> int | None:
		s = (raw or "").strip()
		if not s:
			return None
		# Remove obvious currency artifacts.
		s = re.sub(r"(?i)(₹|rs\.?|inr)", " ", s)
		s = normalize_spaces(s)
		# Handle decimal paise (e.g., 550,000.00)
		m = re.match(r"^([\d,]+)\.(\d{2})$", s)
		if m:
			int_part = digits_only(m.group(1))
			if not int_part:
				return None
			try:
				return int(int_part)
			except Exception:
				return None
		# Otherwise plain integer-ish
		d = digits_only(s)
		if not d:
			return None
		try:
			return int(d)
		except Exception:
			return None

	def _word_to_int(word: str) -> int | None:
		w = (word or "").strip().upper()
		# Common OCR: "S/X" for "SIX"
		w = w.replace("/", "I")
		w = re.sub(r"[^A-Z]", "", w)
		if not w:
			return None
		map_ = {
			"ZERO": 0,
			"ONE": 1,
			"TWO": 2,
			"THREE": 3,
			"FOUR": 4,
			"FIVE": 5,
			"SIX": 6,
			"SEVEN": 7,
			"EIGHT": 8,
			"NINE": 9,
			"TEN": 10,
			"ELEVEN": 11,
			"TWELVE": 12,
			"THIRTEEN": 13,
			"FOURTEEN": 14,
			"FIFTEEN": 15,
			"SIXTEEN": 16,
			"SEVENTEEN": 17,
			"EIGHTEEN": 18,
			"NINETEEN": 19,
			"TWENTY": 20,
			"THIRTY": 30,
			"FORTY": 40,
			"FIFTY": 50,
			"SIXTY": 60,
			"SEVENTY": 70,
			"EIGHTY": 80,
			"NINETY": 90,
		}
		if w in map_:
			return map_[w]
		# Compose like "SIXTYFIVE"
		for tens in ("NINETY", "EIGHTY", "SEVENTY", "SIXTY", "FIFTY", "FORTY", "THIRTY", "TWENTY"):
			if w.startswith(tens) and w[len(tens) :] in map_:
				return map_[tens] + map_[w[len(tens) :]]
		return None

	tractor_lines: List[str] = []
	model_hint_lines: List[str] = []

	from utils.text import parse_hp

	for ln in lines:
		if dealer_name is None:
			if re.search(r"(?i)\bauthori[sz]ed\s+dealer\b", ln):
				dealer_name = _normalize_dealer_line(ln)
			elif dealer_re.search(ln):
				# Avoid common false positives like "Authorized Dealer For ...".
				if not re.search(r"(?i)\bauthori[sz]ed\s+dealer\b", ln):
					dealer_name = _normalize_dealer_line(ln)
		if model_re.search(ln):
			tractor_lines.append(_clean_line(ln))
		# Some docs include model lines without the word "tractor" but with (HP- xx).
		if re.search(r"(?i)\bHP\b", ln) and re.search(r"[A-Za-z]", ln) and re.search(r"\d", ln):
			model_hint_lines.append(_clean_line(ln))
		if horse_power is None:
			# Prefer the shared parser (handles more variants), then fall back to regex.
			hv = parse_hp(ln)
			if hv is not None:
				horse_power = hv
			else:
				m = hp_re.search(ln) or hp_rev_re.search(ln)
				if m:
					horse_power = _normalize_hp(m.group(1))

	# Choose model line: prefer a Tractor line that doesn't look like a pricing line.
	if tractor_lines:
		brands_re = re.compile(
			r"(?i)\b(swaraj|mahindra|massey|ferguson|new\s+holland|sonalika|farmtrac|john\s+deere|kubota|eicher)\b"
		)
		model_token_re = re.compile(r"\b[A-Z]{1,4}\d{2,5}[A-Z0-9\-/+]*\b|\b\d{2,5}\b")
		non_model_re = re.compile(
			r"(?i)\b(servicing|service|free|materials|guarantee|terms|conditions|subsidy|permit|department|delivery)\b"
		)
		accessory_re = re.compile(
			r"(?i)\b(hood|hitch|trailer|trailor|plough|disc\.?\s*plough|tiller|leveller|cage\s*wheel|wheel|toolskit|battery)\b"
		)

		def _model_score(ln: str) -> tuple[int, int, int]:
			penalty = 0
			# "cost of new tractor" lines are often the closest thing to a model line
			# in some templates, so treat them as slightly better than other tractor mentions.
			if re.search(r"(?i)\bcost\b", ln):
				penalty += 1
			if re.search(r"(?i)\bcost\s+of\s+new\b", ln) and re.search(r"(?i)\btractor\b", ln):
				penalty -= 2
			if re.search(r"(?i)\b(?:rs|inr|₹)\b", ln):
				penalty += 2
			if re.search(r"(?i)\bHP\b", ln):
				penalty += 1
			if non_model_re.search(ln):
				penalty += 4
			if accessory_re.search(ln):
				penalty += 4
			moneyish = 1 if money_re.search(ln) else 0
			has_model_token = 0 if (brands_re.search(ln) or model_token_re.search(ln)) else 1
			# Lower tuple is better.
			return (penalty, has_model_token, moneyish)
		model_name = sorted(tractor_lines, key=_model_score)[0]
	elif model_hint_lines:
		# Fall back to an HP-bearing line (often includes model).
		model_name = sorted(model_hint_lines, key=len)[0]

	ignore_money_line_re = re.compile(
		r"(?i)\b(ph\.?|phone|mob\.?|mobile|mo\.?|tel\.?|tin\b|gst\b|pan\b|a/c\b|ac\b|account|ifsc|bank|pin\b|pincode)\b"
	)
	val_meta: List[tuple[int, int, bool]] = []
	for ln in lines:
		if ignore_money_line_re.search(ln):
			continue
		for raw in money_re.findall(ln):
			v = _parse_money_token(raw)
			if v is None or v <= 0:
				continue
			d = digits_only(str(v))
			val_meta.append((int(v), len(d), "," in str(raw)))

	# Amount-in-words fallback: "SIX LAKH SIXTY THOUSAND" => 660000
	# OCR often inserts punctuation (e.g., "SIXTY. THOUSAND"), so normalize.
	words_text = normalize_spaces(text)
	words_norm = re.sub(r"[^A-Za-z0-9/ ]+", " ", (words_text or "").upper())
	words_norm = normalize_spaces(words_norm)
	# Allow a small amount of OCR junk between tokens (layout line breaks, etc.).
	m = re.search(
		r"\b([A-Z0-9/]+)\s+LAKH\s+([A-Z0-9/]+)(?:\s+[A-Z0-9/]+){0,6}\s+THOUSAND\b",
		words_norm,
	)
	if m:
		lakh_raw, thou_raw = m.group(1), m.group(2)
		lakh = None
		thou = None
		if digits_only(lakh_raw):
			try:
				lakh = int(digits_only(lakh_raw))
			except Exception:
				lakh = None
		else:
			lakh = _word_to_int(lakh_raw)
		if digits_only(thou_raw):
			try:
				thou = int(digits_only(thou_raw))
			except Exception:
				thou = None
		else:
			thou = _word_to_int(thou_raw)
		if lakh is not None and thou is not None and 0 <= lakh <= 99 and 0 <= thou <= 99:
			val = int(lakh * 100000 + thou * 1000)
			if val > 0:
				val_meta.append((val, len(str(val)), False))

	if val_meta:
		# Prefer values that look like rupee prices (avoid huge ID-like numbers).
		preferred = [m for m in val_meta if 4 <= m[1] <= 9]
		if preferred:
			comma_pref = [m for m in preferred if m[2]] or preferred
			asset_cost = max(comma_pref, key=lambda t: t[0])[0]
		else:
			# Fallback: pick the largest among reasonably-sized numbers.
			reasonable = [m for m in val_meta if m[1] <= 9]
			asset_cost = max(reasonable, key=lambda t: t[0])[0] if reasonable else None

	# Model-aware HP correction for common OCR confusions.
	# Example: "SWARAJ 744 FE ... 443 HP" is frequently a misread of "48 HP".
	try:
		m_norm = normalize_spaces(re.sub(r"[^A-Za-z0-9 ]+", " ", str(model_name or "")).upper())
		if re.search(r"\bSWARAJ\b", m_norm) and re.search(r"\b744\b", m_norm):
			# If HP was missing or came from a 3-digit noise normalization (e.g. 443->43), prefer 48.
			if horse_power is None or horse_power in {43, 44}:
				horse_power = 48
	except Exception:
		pass

	return {
		"dealer_name": dealer_name,
		"model_name": model_name,
		"horse_power": horse_power,
		"asset_cost": asset_cost,
	}


def detect_stamp_presence(ocr_text: str) -> bool:
	text = (ocr_text or "").lower()
	# Avoid a common false positive in legal docs.
	if "stamp duty" in text:
		return False
	# Require stamp/seal-ish terms.
	return bool(re.search(r"\b(stamp|seal|rubber\s+stamp|company\s+seal)\b", text))


def detect_signature_presence(ocr_text: str) -> bool:
	text = (ocr_text or "").lower()
	# Be conservative: the word "signature" often appears as a printed label.
	return bool(
		re.search(r"\b(authori[sz]ed\s+signatory)\b", text)
		or re.search(r"\bsd/\b", text)
		or re.search(r"\bsigned\b", text)
		or re.search(r"\bsignature\s+of\s+(the\s+)?(beneficiary|customer|purchaser|receiver)\b", text)
	)


def normalize_dealer_name(value: str | None) -> str | None:
	if value is None:
		return None
	s = normalize_spaces(str(value)).strip()
	s = re.sub(r"(?i)^the\s+", "", s)
	s = re.sub(r"[\s\|,:;]+$", "", s).strip()
	m = re.search(r"(?i)\bauthori[sz]ed\s+dealer\s+for\s+(.+)$", s)
	if m:
		s = m.group(1).strip()
	m = re.search(r"(?i)\bdealer\s+for\s+(.+)$", s)
	if m:
		s = m.group(1).strip()
	return normalize_spaces(s).strip() or None
