from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EDAStrategy:
	"""Lightweight, judge-safe knobs derived from offline EDA."""

	header_first: bool = False
	keyword_anchored: bool = False
	hindi_detected: bool = False
	upscale_factor: float = 1.0
	ocr_langs: tuple[str, ...] = ("en", "devanagari", "gujarati")


def _truthy_env(name: str) -> bool:
	v = (os.getenv(name) or "").strip().lower()
	return v in {"1", "true", "yes", "y", "on"}


def _default_profile_paths(repo_root: Path) -> list[Path]:
	# Prefer explicit env override
	p = (os.getenv("EDA_PROFILE_PATH") or "").strip()
	paths: list[Path] = []
	if p:
		paths.append(Path(p))
	# Common locations
	paths.extend(
		[
			repo_root / "eda_profile.json",
			repo_root / "outputs" / "eda_profile.json",
			repo_root / "outputs" / "eda" / "eda_profile.json",
		]
	)
	return paths


def load_eda_profile(repo_root: Path) -> Dict[str, Any]:
	"""Load EDA profile JSON if present; returns {} if missing or invalid."""
	for cand in _default_profile_paths(repo_root):
		try:
			path = cand if cand.is_absolute() else (repo_root / cand)
			if not path.exists() or not path.is_file():
				continue
			obj = json.loads(path.read_text(encoding="utf-8"))
			if isinstance(obj, dict):
				return obj
		except Exception:
			continue
	return {}


def strategy_from_profile(profile: Dict[str, Any]) -> EDAStrategy:
	flags = profile.get("flags") if isinstance(profile, dict) else None
	recs = profile.get("recommendations") if isinstance(profile, dict) else None

	def _get_bool(d: Any, k: str, default: bool) -> bool:
		try:
			v = (d or {}).get(k)
			return bool(v) if v is not None else bool(default)
		except Exception:
			return bool(default)

	def _get_float(d: Any, k: str, default: float) -> float:
		try:
			v = (d or {}).get(k)
			return float(v) if v is not None else float(default)
		except Exception:
			return float(default)

	def _get_tuple_str(d: Any, k: str, default: tuple[str, ...]) -> tuple[str, ...]:
		try:
			v = (d or {}).get(k)
			if isinstance(v, (list, tuple)):
				out = tuple(str(x) for x in v if str(x))
				return out if out else default
		except Exception:
			pass
		return default

	# Allow manual forcing via env (useful in judge harness)
	force_header_first = _truthy_env("DOC_AI_HEADER_FIRST")
	force_keyword_anchor = _truthy_env("DOC_AI_KEYWORD_ANCHORED")

	header_first = bool(force_header_first or _get_bool(recs, "header_first", _get_bool(flags, "top_heavy_text", False)))
	keyword_anchored = bool(force_keyword_anchor or _get_bool(recs, "keyword_anchored", _get_bool(flags, "noisy_layout", False)))
	hindi_detected = bool(_get_bool(flags, "hindi_detected", False))
	upscale_factor = float(_get_float(recs, "upscale_factor", 1.0))
	ocr_langs = _get_tuple_str(recs, "ocr_langs", ("en", "devanagari", "gujarati"))

	# Clamp upscale to a sane range
	if upscale_factor < 1.0:
		upscale_factor = 1.0
	if upscale_factor > 2.5:
		upscale_factor = 2.5

	return EDAStrategy(
		header_first=bool(header_first),
		keyword_anchored=bool(keyword_anchored),
		hindi_detected=bool(hindi_detected),
		upscale_factor=float(upscale_factor),
		ocr_langs=tuple(ocr_langs),
	)
