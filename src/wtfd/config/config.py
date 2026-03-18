"""
wtfd.config
===========

Configuration utilities for the wind turbine fault detection project.

This module centralizes:
- project root detection
- YAML configuration loading
- configured path resolution
- directory creation for pipeline outputs

Design Notes
------------
- Low-level modules such as `wtfd.data`, `wtfd.preprocessing`, and
  `wtfd.schema` should remain path-agnostic whenever practical.
- Notebooks and orchestration scripts should use this module to obtain
  project-relative paths and configuration values.
- Paths defined in `config/config.yaml` are treated as relative to the
  project root unless they are already absolute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


# ---------------------------------------------------------------------------
# Project root and config file location
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the project YAML configuration file.

    Parameters
    ----------
    config_path : str | Path | None, optional
        Explicit path to a YAML config file. If omitted, the default
        project config path is used.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration file is empty.
    """
    path = Path(config_path) if config_path is not None else CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    return config


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_path(path_str: str | Path) -> Path:
    """
    Resolve a config-defined path relative to the project root.

    If the provided path is already absolute, it is returned as-is
    (after normalization). Otherwise, it is interpreted as relative
    to the project root.

    Parameters
    ----------
    path_str : str | Path
        Path string from configuration or code.

    Returns
    -------
    pathlib.Path
        Absolute resolved path.
    """
    path = Path(path_str)

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def get_path(path_key: str, config: dict[str, Any] | None = None) -> Path:
    """
    Retrieve and resolve a configured path from the 'paths' section.

    Parameters
    ----------
    path_key : str
        Key under config['paths'].
    config : dict[str, Any] | None, optional
        Loaded configuration dictionary. If omitted, the config is loaded.

    Returns
    -------
    pathlib.Path
        Absolute resolved path for the requested key.

    Raises
    ------
    KeyError
        If the key is not present in the 'paths' section.
    """
    cfg = config if config is not None else load_config()
    paths = cfg.get("paths", {})

    if path_key not in paths:
        available = ", ".join(sorted(paths.keys()))
        raise KeyError(
            f"Path key '{path_key}' not found in config['paths']. "
            f"Available keys: {available}"
        )

    return resolve_path(paths[path_key])


def get_all_paths(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """
    Resolve all configured paths from the 'paths' section.

    Parameters
    ----------
    config : dict[str, Any] | None, optional
        Loaded configuration dictionary. If omitted, the config is loaded.

    Returns
    -------
    dict[str, pathlib.Path]
        Dictionary mapping path keys to resolved absolute paths.
    """
    cfg = config if config is not None else load_config()
    paths = cfg.get("paths", {})

    return {key: resolve_path(value) for key, value in paths.items()}


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def ensure_directories(
    config: dict[str, Any] | None = None,
    path_keys: list[str] | None = None,
) -> None:
    """
    Create configured directories if they do not already exist.

    This is intended for project-controlled directories such as:
    - interim outputs
    - processed data
    - models
    - logs

    Parameters
    ----------
    config : dict[str, Any] | None, optional
        Loaded configuration dictionary. If omitted, the config is loaded.
    path_keys : list[str] | None, optional
        Specific path keys to create. If omitted, all configured paths
        are created.

    Notes
    -----
    Creating all configured paths is usually fine for this project because
    the raw-data paths are directories as well. This function does not
    delete or overwrite anything.
    """
    cfg = config if config is not None else load_config()
    resolved_paths = get_all_paths(cfg)

    keys_to_create = path_keys if path_keys is not None else list(resolved_paths.keys())

    for key in keys_to_create:
        if key not in resolved_paths:
            raise KeyError(f"Path key '{key}' not found in config['paths'].")

        resolved_paths[key].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Convenience getters for commonly used config values
# ---------------------------------------------------------------------------

def get_project_name(config: dict[str, Any] | None = None) -> str:
    """
    Get the configured project name.
    """
    cfg = config if config is not None else load_config()
    return str(cfg.get("project", {}).get("name", "wtfd"))


def get_random_seed(config: dict[str, Any] | None = None) -> int:
    """
    Get the configured project random seed.

    Falls back to the model random_state if needed.
    """
    cfg = config if config is not None else load_config()

    if "random_seed" in cfg.get("project", {}):
        return int(cfg["project"]["random_seed"])

    if "random_state" in cfg.get("model", {}):
        return int(cfg["model"]["random_state"])

    return 1984


def get_logging_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Get the logging configuration block.
    """
    cfg = config if config is not None else load_config()
    return dict(cfg.get("logging", {}))


def get_data_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Get the data configuration block.
    """
    cfg = config if config is not None else load_config()
    return dict(cfg.get("data", {}))


def get_model_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Get the model configuration block.
    """
    cfg = config if config is not None else load_config()
    return dict(cfg.get("model", {}))