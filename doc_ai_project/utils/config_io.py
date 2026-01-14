from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict

from utils.config import PipelineConfig


def pipeline_config_to_dict(cfg: PipelineConfig) -> Dict[str, Any]:
	"""Serialize PipelineConfig to JSON-friendly dict."""
	d = asdict(cfg)
	# normalize Path -> str
	for k, v in list(d.items()):
		if isinstance(v, Path):
			d[k] = str(v)
		elif isinstance(v, tuple):
			# keep tuples JSON-friendly
			d[k] = list(v)
	return d


def pipeline_config_from_dict(data: Dict[str, Any]) -> PipelineConfig:
	"""Best-effort dict -> PipelineConfig conversion.

	Unknown keys are ignored so older/newer configs remain compatible.
	"""
	allowed = {f.name: f for f in fields(PipelineConfig)}
	kwargs: Dict[str, Any] = {}
	for k, v in (data or {}).items():
		if k not in allowed:
			continue
		ft = allowed[k].type
		# Path fields
		if k.endswith("_dir") or k.endswith("_path"):
			if v in (None, ""):
				kwargs[k] = None if "| None" in str(ft) else Path("outputs")
			else:
				kwargs[k] = Path(v)
			continue
		# Tuple fields: accept list
		if isinstance(getattr(PipelineConfig, k, None), tuple) and isinstance(v, list):
			kwargs[k] = tuple(v)
			continue
		kwargs[k] = v
	return PipelineConfig(**kwargs)


def load_config_json(path: Path) -> PipelineConfig:
	obj = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(obj, dict):
		raise ValueError("config JSON must be an object")
	return pipeline_config_from_dict(obj)


def save_config_json(cfg: PipelineConfig, path: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(pipeline_config_to_dict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
	return path
