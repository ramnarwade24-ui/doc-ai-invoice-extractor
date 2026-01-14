from __future__ import annotations

import re
from typing import Iterable


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")
LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_spaces(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


PUNCT_RE = re.compile(r"[^\w\s\u0900-\u097F\u0A80-\u0AFF]+", re.UNICODE)


def normalize_name(text: str) -> str:
	"""Normalize entity-like names for matching: lowercase, strip punctuation, collapse spaces."""
	if not text:
		return ""
	t = text.strip().lower()
	t = PUNCT_RE.sub(" ", t)
	t = normalize_spaces(t)
	return t


HP_NORM_RE = re.compile(r"(?i)(?P<num>\d{1,3})\s*(?:h\.?\s*p\.?|horse\s*power|एच\.?\s*पी\.?|હો\.?\s*પી\.?)")


# Canonical keyword normalization for Hindi/Gujarati + common OCR variants.
_KEYWORD_CANON = {
	# model
	"मॉडल": "model",
	"माडल": "model",
	"मोडल": "model",
	"મોડલ": "model",
	"મોડેલ": "model",
	# horsepower
	"एचपी": "hp",
	"एच.पी": "hp",
	"એચપી": "hp",
	"એચ.પી": "hp",
	"હોર્સ પાવર": "hp",
	"હૉર્સ પાવર": "hp",
	"horse power": "hp",
	"horsepower": "hp",
	# amount/price
	"राशि": "amount",
	"कुल": "total",
	"कुल राशि": "total",
	"कुल रकम": "total",
	"किंमत": "amount",
	"કુલ": "total",
	"રકમ": "amount",
	"કિંમત": "amount",
	"ટોટલ": "total",
}


def normalize_keywords(text: str) -> str:
	"""Replace key Hindi/Gujarati tokens with canonical English ones.

	This is not full transliteration; it only targets extraction keywords.
	"""
	if not text:
		return ""
	t = normalize_spaces(text)
	# deterministic replacements; do longest-first
	for src in sorted(_KEYWORD_CANON.keys(), key=len, reverse=True):
		t = re.sub(re.escape(src), _KEYWORD_CANON[src], t, flags=re.IGNORECASE)
	return t


def parse_hp(text: str) -> int | None:
	"""Extract horsepower integer from noisy variants like '50 HP', '50 H.P.', '50hp'."""
	if not text:
		return None
	# normalize keyword variants to make regex more robust
	t = normalize_keywords(text)
	m = HP_NORM_RE.search(t)
	if not m:
		return None
	try:
		v = int(m.group("num"))
	except Exception:
		return None
	if v <= 0 or v > 300:
		return None
	return v


CURRENCY_TOKENS_RE = re.compile(r"(?i)(₹|rs\.?|inr)")


def strip_currency_tokens(text: str) -> str:
	return CURRENCY_TOKENS_RE.sub(" ", text or "")


def digits_only(text: str) -> str:
	return re.sub(r"\D+", "", text)


def detect_language_bucket(texts: Iterable[str]) -> str:
	"""Heuristic language bucket for EDA: 'en' | 'hi' | 'gu' | 'mixed'."""
	joined = " ".join([t for t in texts if t])
	if not joined:
		return "unknown"

	dev = len(DEVANAGARI_RE.findall(joined))
	guj = len(GUJARATI_RE.findall(joined))
	lat = len(LATIN_RE.findall(joined))

	# Pick dominant script
	mx = max(dev, guj, lat)
	if mx == 0:
		return "unknown"

	# Mixed if second-best is close
	sorted_counts = sorted([dev, guj, lat], reverse=True)
	if sorted_counts[1] / sorted_counts[0] >= 0.35:
		return "mixed"

	if mx == lat:
		return "en"
	if mx == dev:
		return "hi"
	return "gu"
