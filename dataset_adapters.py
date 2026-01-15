#!/usr/bin/env python3
"""Dataset-aware adapters that learn normalization maps and inject into extraction.

This script analyzes dataset runs (predictions + optional labels) to learn:
- Common dealer OCR patterns (character confusions, spacing/punctuation patterns)
- Model naming normalization (hyphens, spacing, roman/series tokens)
- Price formatting styles (separator normalization)

It writes a JSON artifact consumed dynamically by doc_ai_project/extractor.py:
- doc_ai_project/outputs/dataset_adapters.json

Deterministic: all iterations are sorted and any sampling is seeded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _doc_ai_dir(repo_root: Path) -> Path:
	return repo_root / "doc_ai_project"


def _load_json(path: Path) -> Dict[str, Any]:
	obj = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(obj, dict):
		raise ValueError(f"Expected JSON object at {path}")
	return obj


def _normalize_name(text: str) -> str:
	repo_root = _repo_root()
	import sys
	import importlib

	sys.path.insert(0, str(_doc_ai_dir(repo_root)))
	utils_text = importlib.import_module("utils.text")
	return utils_text.normalize_name(text or "")


def _fuzzy(a: str, b: str) -> float:
	from rapidfuzz import fuzz

	return float(fuzz.token_set_ratio(_normalize_name(a), _normalize_name(b)))


@dataclass(frozen=True)
class Pair:
	pred: str
	gt: str


def _load_pairs(*, pred_dir: Path, labels_dir: Optional[Path]) -> List[Pair]:
	pairs: List[Pair] = []
	if labels_dir is None or not labels_dir.exists():
		return pairs

	for pred_path in sorted(pred_dir.glob("*.json")):
		doc_id = pred_path.stem
		gt_path = labels_dir / f"{doc_id}.json"
		if not gt_path.exists():
			continue
		try:
			pred_obj = json.loads(pred_path.read_text(encoding="utf-8"))
			gt_obj = json.loads(gt_path.read_text(encoding="utf-8"))
		except Exception:
			continue
		pred_fields = (pred_obj.get("fields") or {}) if isinstance(pred_obj, dict) else {}
		if isinstance(gt_obj, dict) and "fields" in gt_obj and isinstance(gt_obj["fields"], dict):
			gt_fields = gt_obj["fields"]
		elif isinstance(gt_obj, dict):
			gt_fields = gt_obj
		else:
			continue

		for key in ("dealer_name", "model_name"):
			pv = pred_fields.get(key)
			gv = gt_fields.get(key)
			if pv in (None, "") or gv in (None, ""):
				continue
			pairs.append(Pair(pred=str(pv), gt=str(gv)))

	return pairs


def _candidate_replacements() -> List[Tuple[str, str]]:
	# Common OCR confusions in invoices; ordered deterministically.
	return [
		("0", "o"),
		("1", "l"),
		("|", "l"),
		("5", "s"),
		("8", "b"),
		("rn", "m"),
		("vv", "w"),
		("\u00a0", " "),  # nbsp
	]


def _apply_replacements(text: str, reps: List[Tuple[str, str]]) -> str:
	out = text or ""
	for a, b in reps:
		out = out.replace(a, b)
	return out


def _learn_replacements(pairs: List[Pair], *, min_gain: float = 0.6, max_rules: int = 8) -> List[Tuple[str, str]]:
	"""Greedy select replacements that improve fuzzy similarity on near-miss pairs."""
	if not pairs:
		return []

	# Focus on near-miss: high-ish similarity but not exact
	near = [p for p in pairs if 70.0 <= _fuzzy(p.pred, p.gt) < 99.0]
	if not near:
		return []

	available = _candidate_replacements()
	selected: List[Tuple[str, str]] = []

	def avg_score(rules: List[Tuple[str, str]]) -> float:
		s = 0.0
		for pr in near:
			s += _fuzzy(_apply_replacements(pr.pred, rules), pr.gt)
		return float(s / max(1, len(near)))

	base = avg_score([])
	current = base

	for _ in range(max_rules):
		best_rule = None
		best_score = current
		for r in available:
			if r in selected:
				continue
			score = avg_score(selected + [r])
			if score > best_score + 1e-9:
				best_score = score
				best_rule = r
		if best_rule is None:
			break
		gain = best_score - current
		if gain < min_gain:
			break
		selected.append(best_rule)
		current = best_score

	return selected


def _price_style() -> Dict[str, Any]:
	# This is intentionally conservative and deterministic; it just normalizes separators.
	return {
		"normalize_separators": True,
		"allowed_separators": [",", ".", " "],
	}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Learn dataset adapters and write dataset_adapters.json")
	p.add_argument("--labels", default="", help="Optional labels folder (repo-root relative)")
	p.add_argument(
		"--pred-dir",
		default="doc_ai_project/outputs/eval_predictions",
		help="Predictions folder to learn from (repo-root relative)",
	)
	p.add_argument(
		"--out",
		default="doc_ai_project/outputs/dataset_adapters.json",
		help="Output adapters JSON path (repo-root relative)",
	)
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = _repo_root()

	pred_dir = repo_root / args.pred_dir
	labels_dir = (repo_root / args.labels) if args.labels else None
	out_path = repo_root / args.out

	pairs = _load_pairs(pred_dir=pred_dir, labels_dir=labels_dir)

	# Learn replacements separately for dealer/model but share the same rule list for simplicity.
	replacements = _learn_replacements(pairs)

	adapters = {
		"version": 1,
		"dealer": {
			"replacements": replacements,
			"note": "Applied before fuzzy matching and normalization.",
		},
		"model": {
			"replacements": replacements,
			"strip_punct": True,
			"collapse_spaces": True,
		},
		"price": _price_style(),
		"stats": {
			"pairs_used": int(len(pairs)),
			"rules_selected": int(len(replacements)),
		},
	}

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(adapters, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps({"out": str(out_path), "rules": replacements}, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
