from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


CONFIG: Dict[str, Any] = {}


def load_config(config_path: str = None) -> Dict[str, Any]:
    global CONFIG
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "config.yaml")
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config = _resolve_env_vars(config)
    CONFIG.update(config)
    return CONFIG


def _resolve_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_var = obj[2:-1]
        return os.getenv(env_var, "")
    return obj


def setup_logging():
    log_config = CONFIG.get("logging", {})
    level = log_config.get("level", "INFO")
    log_file = log_config.get("file", "logs/scanner.log")
    fmt = log_config.get(
        "format", "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}"
    )
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        colorize=True,
    )
    logger.add(
        log_file,
        level=level,
        format=fmt,
        rotation="10 MB",
        retention="5 days",
    )
    logger.info(f"Logging configured: level={level}, file={log_file}")


load_config()
setup_logging()