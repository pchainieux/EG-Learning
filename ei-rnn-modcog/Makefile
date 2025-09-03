SHELL := /bin/bash
export PYTHONPATH := $(PWD)

# Use '>' instead of TAB to start recipe lines (avoids "missing separator")
.RECIPEPREFIX := >

PY ?= python -u
PYTHON ?= $(PY)

# =========================================================
# EI GRADIENT EXPERIMENT
# =========================================================

CFG ?= configs/ei_gradient/credit.yaml

.PHONY: eigrad-collect eigrad-metrics eigrad-plots eigrad-all

eigrad-collect:
> $(PYTHON) -m experiments.ei_gradient.scripts.collect_grads --cfg $(CFG)

eigrad-metrics:
> $(PYTHON) -m experiments.ei_gradient.scripts.compute_metrics --cfg $(CFG)

eigrad-plots:
> $(PYTHON) -m experiments.ei_gradient.scripts.plot_all --cfg $(CFG)

eigrad-all: eigrad-collect eigrad-metrics eigrad-plots

# ================================
# SIMPLE FIXED-POINTS PIPELINE
# ================================

# Main config (YAML) and output locations
CONFIG ?= fixed_points.yaml
OUT_ROOT ?= outputs/fp_runs
NAME ?= default
OUTDIR ?= $(OUT_ROOT)/$(NAME)

# Grid defaults (edit freely)
INH_LEVELS ?= 0.10 0.20 0.35 0.50
OPTIMS ?= eg gd
SEEDS ?= 1 2 3

.PHONY: help fp-all fp-one train cache-build fp-grid aggregate clean

help:
> @echo "Targets:"
> @echo "  EI gradient:  eigrad-collect eigrad-metrics eigrad-plots eigrad-all [CFG=...]"
> @echo "  Fixed points: fp-all fp-one train cache-build fp-grid aggregate clean"
> @echo
> @echo "Vars:"
> @echo "  CONFIG=<cfg.yaml> OUTDIR=<outputs/fp_runs/your_name>"
> @echo "  OUT_ROOT, OPTIMS='eg gd', INH_LEVELS='0.10 0.20 0.35 0.50', SEEDS='1 2 3'"

# Train (uses CONFIG)
train:
> $(PY) train_singlehead_modcog.py --config $(CONFIG)

# Full fixed-point pipeline for a single run (train if needed, analyze, plot)
fp-all: fp-one
fp-one:
> $(PY) -m experiments.fixed_points.scripts.run_fixed_points --config $(CONFIG) --outdir $(OUTDIR) --plot

# Pre-build cache files according to CONFIG.data.cache.{num_batches, paths, seed, ...}
cache-build:
> $(PY) - <<'PY'
> import yaml, sys
> from pathlib import Path
> from src.data.dataset_cached import build_cached_dataset, save_cached_npz
> cfg = yaml.safe_load(open("$(CONFIG)", "r")) or {}
> data = cfg.get("data", {})
> if data.get("source", "online").lower() != "cached":
>     print("data.source is not 'cached' in", "$(CONFIG)"); sys.exit(0)
> cache = data.get("cache", {}) or {}
> nb = int(cache.get("num_batches", 200))
> bs = int(data.get("batch_size", 128))
> L  = int(data.get("seq_len", 80))
> seed = cache.get("seed", None)
> paths = cache.get("paths", {}) or {}
> tasks = cfg.get("tasks", ["dm1"])
> if isinstance(tasks, str): tasks = [tasks]
> for t in tasks:
>     out = paths.get(t, f"runs/cache/{t}_{nb}x{bs}x{L}.npz")
>     Path(out).parent.mkdir(parents=True, exist_ok=True)
>     cached, _ = build_cached_dataset(t, nb, bs, L, seed=seed)
>     save_cached_npz(out, cached, meta={"task": t, "seq_len": L})
>     print("wrote", out)
> PY

# Grid over optimizer x inhibitory fraction x seed:
# creates a temporary config per combo and runs the full pipeline.
fp-grid:
> set -euo pipefail; \
> mkdir -p "$(OUT_ROOT)"; \
> for opt in $(OPTIMS); do \
>   for inh in $(INH_LEVELS); do \
>     for s in $(SEEDS); do \
>       out="$(OUT_ROOT)/opt-$${opt}/inh-$${inh}/seed-$${s}"; \
>       cfg="$$out/config.yaml"; \
>       mkdir -p "$$out/run"; \
>       $(PY) -u -c 'import sys, yaml, pathlib as p; \
> cfg_in, cfg_out, out, opt, inh, seed = sys.argv[1:7]; \
> cfg = yaml.safe_load(open(cfg_in, "r")) or {}; \
> cfg.setdefault("optim", {})["algorithm"] = opt; \
> cfg["model"] = cfg.get("model", {}); \
> cfg["model"]["exc_frac"] = 1.0 - float(inh); \
> cfg["seed"] = int(float(seed)); \
> cfg["outdir"] = out; \
> p.Path(out).mkdir(parents=True, exist_ok=True); \
> yaml.safe_dump(cfg, open(cfg_out, "w"), sort_keys=False)' \
>         "$(CONFIG)" "$$cfg" "$$out" "$$opt" "$$inh" "$$s"; \
>       $(PY) run_fixed_points.py --config "$$cfg" --outdir "$$out" --plot; \
>     done; \
>   done; \
> done

# Aggregate per-run summaries into a single CSV under OUT_ROOT
aggregate:
> $(PY) aggregate.py --root $(OUT_ROOT) --out $(OUT_ROOT)/aggregate.csv
> @echo "aggregate → $(OUT_ROOT)/aggregate.csv"

# Nuke the OUT_ROOT directory (be careful!)
clean:
> rm -rf $(OUT_ROOT)
