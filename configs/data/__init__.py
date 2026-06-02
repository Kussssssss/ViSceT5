"""configs/data/__init__.py — Dataset YAML config loader."""
import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

DATA_CONFIG_DIR = Path(__file__).parent

def load_dataset_config(name_or_path: str) -> Dict[str, Any]:
    """Load a dataset YAML config by name or path."""
    if os.path.isfile(name_or_path):
        path = Path(name_or_path)
    else:
        path = DATA_CONFIG_DIR / f"{name_or_path}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def list_available_datasets() -> List[str]:
    """List all dataset YAML configs in configs/data/."""
    # List all .yaml files except Vocab.yaml which is for vocabs
    return [p.stem for p in DATA_CONFIG_DIR.glob("*.yaml") if p.stem != "Vocab"]

def resolve_source(cfg_section: Dict) -> tuple:
    """Returns (local_dir_or_None, drive_id_or_None)."""
    local_dir = cfg_section.get('dir', '')
    drive_id = cfg_section.get('drive_id', '')
    if local_dir and os.path.isdir(local_dir):
        return local_dir, None
    return None, drive_id if drive_id else None
