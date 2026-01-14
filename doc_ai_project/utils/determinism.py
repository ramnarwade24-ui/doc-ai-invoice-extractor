from __future__ import annotations

import os
import random
from typing import Optional


def set_deterministic(seed: int = 1337, deterministic: bool = True) -> None:
	"""Best-effort determinism across common libs.

	- Always seeds Python + common env vars
	- Seeds numpy if installed
	- Seeds torch if installed

	This project is CPU-first; GPU determinism is out of scope.
	"""
	seed = int(seed)

	os.environ.setdefault("PYTHONHASHSEED", str(seed))
	random.seed(seed)

	try:
		import numpy as np  # type: ignore

		np.random.seed(seed)
	except Exception:
		pass

	try:
		import torch  # type: ignore

		torch.manual_seed(seed)
		if hasattr(torch, "use_deterministic_algorithms"):
			torch.use_deterministic_algorithms(bool(deterministic))
		# CPU threads can affect reproducibility slightly; keep it stable.
		if deterministic:
			torch.set_num_threads(1)
	except Exception:
		pass
