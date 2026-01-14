from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def generate_pseudo_label(doc_id: str, extracted_fields: Dict[str, object], confidence: float) -> Dict[str, object]:
	"""Pseudo-labels for no-ground-truth scenarios.

	Strategy:
	- Keep only high-confidence fields
	- Store doc-level confidence for later filtering
	"""
	keep = {}
	for k, v in extracted_fields.items():
		if v in (None, ""):
			continue
		keep[k] = v

	return {
		"doc_id": doc_id,
		"pseudo_fields": keep,
		"doc_confidence": confidence,
		"note": "pseudo-label (rule-based)"
	}


def write_pseudo_label(out_path: Path, label: Dict[str, object]) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
