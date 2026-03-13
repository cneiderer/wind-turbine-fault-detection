from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    """Load the YAML project configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories() -> None:
    """Create configured project directories if they do not already exist."""
    config = load_config()

    for path_str in config.get("paths", {}).values():
        (PROJECT_ROOT / path_str).mkdir(parents=True, exist_ok=True)