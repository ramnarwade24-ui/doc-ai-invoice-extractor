from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger


def get_json_logger(name: str, log_path: Optional[Path] = None) -> logging.Logger:
	logger = logging.getLogger(name)
	if getattr(logger, "_configured", False):
		return logger

	logger.setLevel(logging.INFO)
	logger.propagate = False

	formatter = jsonlogger.JsonFormatter(
		fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
		rename_fields={"levelname": "level", "name": "logger"},
	)

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
