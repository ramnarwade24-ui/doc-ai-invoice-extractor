from __future__ import annotations

from pathlib import Path
from typing import List

import csv


DEFAULT_DEALERS = [
	"ABC Tractors Pvt Ltd",
	"XYZ Motors",
	"Shree Tractors",
	"Patel Agro",
	"IDFC FIRST Bank Empanelled Dealer",
]

DEFAULT_MODELS = [
	"Mahindra 575 DI",
	"Mahindra 265 DI",
	"Swaraj 735 FE",
	"Sonalika DI 745 III",
	"John Deere 5045D",
]


def _load_lines(path: Path) -> List[str]:
	if not path.exists():
		return []
	lines = []
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if line and not line.startswith("#"):
			lines.append(line)
	return lines


def _load_csv_first_col(path: Path) -> List[str]:
	if not path.exists():
		return []
	rows: List[str] = []
	with path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		for row in reader:
			if not row:
				continue
			cell = (row[0] or "").strip()
			if not cell:
				continue
			# skip header-like cells
			if cell.lower() in {"dealer_name", "dealer", "name", "model_name", "model"}:
				continue
			rows.append(cell)
	return rows


def load_dealer_master_list(base_dir: Path) -> List[str]:
	# Allow user to provide their own list.
	for p in [base_dir / "data" / "dealers_master_list.txt", base_dir / "data" / "dealers_master_list.csv"]:
		lines = _load_csv_first_col(p) if p.suffix.lower() == ".csv" else _load_lines(p)
		if lines:
			return lines
	return DEFAULT_DEALERS


def load_model_master_list(base_dir: Path) -> List[str]:
	for p in [base_dir / "data" / "models_master_list.txt", base_dir / "data" / "models_master_list.csv"]:
		lines = _load_csv_first_col(p) if p.suffix.lower() == ".csv" else _load_lines(p)
		if lines:
			return lines
	return DEFAULT_MODELS
