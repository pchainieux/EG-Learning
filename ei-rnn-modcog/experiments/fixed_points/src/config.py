from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f) or {}

def deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_yaml(path)
    if overrides:
        deep_update(cfg, overrides)
    return cfg


def get_outdir(cfg: Mapping[str, Any]) -> Path:
    outdir = cfg.get("outdir", "runs/exp")
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p
