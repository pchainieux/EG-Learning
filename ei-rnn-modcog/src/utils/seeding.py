from __future__ import annotations
import os
import random
import numpy as np
import torch

def set_seed_all(seed: int, *, deterministic: bool = False) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)

def device_name(d: torch.device) -> str:
    if d.type == "cuda":
        idx = d.index or 0
        name = torch.cuda.get_device_name(idx)
        return f"cuda:{idx} ({name})"
    return "cpu"
