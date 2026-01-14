from __future__ import annotations

from typing import Optional

from utils.text import digits_only, normalize_spaces


def clean_str(value: Optional[str]) -> Optional[str]:
	if value is None:
		return None
	v = normalize_spaces(value)
	return v if v else None


def clean_int(value: Optional[int | str]) -> Optional[int]:
	if value is None:
		return None
	if isinstance(value, int):
		return value
	d = digits_only(str(value))
	return int(d) if d else None
