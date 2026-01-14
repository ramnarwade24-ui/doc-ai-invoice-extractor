from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed() -> float:
	start = time.perf_counter()
	value = {"sec": 0.0}
	try:
		yield value
	finally:
		value["sec"] = time.perf_counter() - start
