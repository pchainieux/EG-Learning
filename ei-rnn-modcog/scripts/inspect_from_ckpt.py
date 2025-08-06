# python -m scripts.inspect_from_ckpt

import torch, yaml
from pathlib import Path
from src.models.ei_rnn import EIRNN, EIConfig
from src.analysis import inspect_weights as iw

ckpt_path = Path("experiments/multihead_epoch015.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

cfg = ckpt["config"]
hidden = int(cfg["model"]["hidden_size"])
exc_frac = float(cfg["model"]["exc_frac"])
spectral_radius = float(cfg["model"]["spectral_radius"])
input_scale = float(cfg["model"]["input_scale"])
input_dim = int(cfg["data"]["input_dim"]) if "input_dim" in cfg.get("data", {}) else 33 

core = EIRNN(input_dim, output_size=max(ckpt["head_dims"].values()), 
             cfg=EIConfig(hidden_size=hidden, exc_frac=exc_frac,
                          spectral_radius=spectral_radius, input_scale=input_scale))
core.load_state_dict(ckpt["core"], strict=False)

outdir = Path("experiments/inspect_from_ckpt/")
iw.save_weight_hist_by_group(core.W_hh, core.sign_vec, str(outdir / "weights_hist_from_ckpt.png"))
iw.save_row_col_sums(core.W_hh, str(outdir / "rowcol_sums_from_ckpt.png"))
iw.save_gram_spectrum(core.W_hh, str(outdir / "spectrum_from_ckpt.png"))
iw.save_matrix_heatmap(core.W_hh, str(outdir / "W_hh_heatmap_from_ckpt.png"))
iw.save_dale_violation_report(core.W_hh, core.sign_vec, str(outdir / "dale_report_from_ckpt.txt"))
