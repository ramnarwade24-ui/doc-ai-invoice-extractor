from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger


def get_json_logger(name: str, log_path: Optional[Path] = None, *, to_stdout: bool = True) -> logging.Logger:
	logger = logging.getLogger(name)
	# If already configured, still honor to_stdout by adding/removing stream handlers.
	if getattr(logger, "_configured", False):
		if not to_stdout:
			# Remove any StreamHandler to keep stdout clean in judge/jury runs.
			for h in list(logger.handlers):
				if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
					logger.removeHandler(h)
		else:
			# Ensure at least one StreamHandler exists if requested.
			has_stream = any(
				isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers
			)
			if not has_stream:
				formatter = jsonlogger.JsonFormatter(
					fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
					rename_fields={"levelname": "level", "name": "logger"},
				)
				stream = logging.StreamHandler()
				stream.setFormatter(formatter)
				logger.addHandler(stream)
		# If log_path is provided later, attach it (idempotent by filepath).
		if log_path is not None:
			try:
				log_path.parent.mkdir(parents=True, exist_ok=True)
				want = str(log_path)
				has_file = any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == want for h in logger.handlers)
				if not has_file:
					formatter = jsonlogger.JsonFormatter(
						fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
						rename_fields={"levelname": "level", "name": "logger"},
					)
					fh = logging.FileHandler(log_path, encoding="utf-8")
					fh.setFormatter(formatter)
					logger.addHandler(fh)
			except Exception:
				pass
		return logger

	logger.setLevel(logging.INFO)
	logger.propagate = False

	formatter = jsonlogger.JsonFormatter(
		fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
		rename_fields={"levelname": "level", "name": "logger"},
	)

	if to_stdout:
		stream = logging.StreamHandler()
		stream.setFormatter(formatter)
		logger.addHandler(stream)

	if log_path is not None:
		log_path.parent.mkdir(parents=True, exist_ok=True)
		fh = logging.FileHandler(log_path, encoding="utf-8")
		fh.setFormatter(formatter)
		logger.addHandler(fh)

	setattr(logger, "_configured", True)
	return logger


def log_event(logger: logging.Logger, event: str, **kwargs: Any) -> None:
	payload: Dict[str, Any] = {"event": event, **kwargs}
	logger.info(payload)
