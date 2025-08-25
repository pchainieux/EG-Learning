from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml
import torch
import pandas as pd
import datetime as dt


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def ensure_outdir(base_out: str | Path, exp_id: str) -> Path:
    d = Path(base_out) / exp_id
    (d / "figs").mkdir(parents=True, exist_ok=True)
    return d


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_tensors(tensors: Dict[str, torch.Tensor], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in tensors.items()}, path)


def load_tensors(path: str | Path) -> Dict[str, torch.Tensor]:
    return torch.load(Path(path), map_location="cpu")


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def default_exp_dir(out_dir: str | Path, exp_id: Optional[str]) -> Path:
    if exp_id is None:
        exp_id = f"exp_{timestamp()}"
    return ensure_outdir(out_dir, exp_id)


def load_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(checkpoint_path), map_location=map_location)
