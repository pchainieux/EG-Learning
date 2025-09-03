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
# FIXED-POINTS PIPELINE (SIMPLIFIED)
# ================================

# Main config (YAML) and output locations
CONFIG ?= configs/fixed_points/fixed_points.yaml
OUT_ROOT ?= outputs/fp_runs
NAME ?= default
OUTDIR ?= $(OUT_ROOT)/$(NAME)

# Grid defaults
INH_LEVELS ?= 0.10 0.20 0.35 0.50
OPTIMS ?= eg gd
SEEDS ?= 1 2 3

.PHONY: help fp-train fp-rollout fp-solve fp-plot fp-single-run fp-grid fp-aggregate fp-clean

help:
> @echo "Fixed Points Pipeline Targets:"
> @echo "  fp-train       : Train a single model (uses CONFIG, OUTDIR)"
> @echo "  fp-rollout     : Collect hidden state seeds from a trained model"
> @echo "  fp-solve       : Find and analyze fixed points from seeds"
> @echo "  fp-plot        : Generate plots from fixed point analysis"
> @echo "  fp-single-run  : Run the full pipeline for a single config (train -> rollout -> solve -> plot)"
> @echo "  fp-grid-config : Generate config files for a parameter grid (does not run them)"
> @echo "  fp-grid        : Run fp-single-run for each config in the grid (WARNING: runs jobs sequentially!)"
> @echo "  fp-aggregate   : Aggregate results from OUT_ROOT into a single CSV"
> @echo "  fp-clean       : Remove the OUT_ROOT directory"
> @echo ""
> @echo "Variables:"
> @echo "  CONFIG=$(CONFIG)"
> @echo "  OUTDIR=$(OUTDIR)"
> @echo "  OUT_ROOT=$(OUT_ROOT)"
> @echo "  OPTIMS=$(OPTIMS)"
> @echo "  INH_LEVELS=$(INH_LEVELS)"
> @echo "  SEEDS=$(SEEDS)"

# 1. Train a model
fp-train:
> $(PYTHON) -m base_scripts.train_singlehead_modcog --config $(CONFIG) --outdir $(OUTDIR)

# 2. Collect rollout seeds (requires a trained model)
fp-rollout:
> $(PYTHON) -m experiments.fixed_points.scripts.run_fixed_points --config $(CONFIG) --outdir $(OUTDIR) --no-train

# 3. Solve for fixed points (requires rollout seeds)
fp-solve:
> $(PYTHON) -m experiments.fixed_points.scripts.unified_fixed_points --run $(OUTDIR)/run --config $(CONFIG)

# 4. Plot results (requires solved fixed points)
fp-plot:
> $(PYTHON) -m experiments.fixed_points.scripts.plot_fixed_points --run $(OUTDIR)/run

# 5. Full pipeline for a single configuration
fp-single-run: fp-train fp-rollout fp-solve fp-plot

# 6. Generate config files for a parameter grid
fp-grid-config:
> @echo "Generating config files for grid in $(OUT_ROOT)..."
> set -euo pipefail; \
> mkdir -p "$(OUT_ROOT)"; \
> for opt in $(OPTIMS); do \
>   for inh in $(INH_LEVELS); do \
>     for s in $(SEEDS); do \
>       out="$(OUT_ROOT)/opt-$${opt}/inh-$${inh}/seed-$${s}"; \
>       cfg="$$out/config.yaml"; \
>       mkdir -p "$$out/run"; \
>       $(PYTHON) -c " \
> import sys, yaml, pathlib as p; \
> cfg_in, cfg_out, out, opt, inh, seed = sys.argv[1:7]; \
> cfg = yaml.safe_load(open(cfg_in, 'r')) or {}; \
> cfg.setdefault('optim', {})['algorithm'] = opt; \
> cfg['model'] = cfg.get('model', {}); \
> cfg['model']['exc_frac'] = 1.0 - float(inh); \
> cfg['seed'] = int(float(seed)); \
> cfg['outdir'] = out; \
> p.Path(out).mkdir(parents=True, exist_ok=True); \
> with open(cfg_out, 'w') as f: \
>     yaml.safe_dump(cfg, f, sort_keys=False) \
>       " "$(CONFIG)" "$$cfg" "$$out" "$$opt" "$$inh" "$$s"; \
>       echo "Generated: $$cfg"; \
>     done; \
>   done; \
> done
> @echo "Config generation complete. Now run: make fp-grid-run"

# 7. Run the full pipeline for each generated config (SEQUENTIAL - for testing)
fp-grid-run:
> @echo "WARNING: This runs jobs SEQUENTIALLY. For a real grid, use a job array."
> set -euo pipefail; \
> for cfg in $$(find "$(OUT_ROOT)" -name "config.yaml"); do \
>   outdir=$$(dirname "$$cfg"); \
>   echo "--- Running pipeline for $$outdir ---"; \
>   $(PYTHON) -m experiments.fixed_points.scripts.run_fixed_points --config "$$cfg" --outdir "$$outdir" --plot || { echo "Failed on $$outdir"; exit 1; }; \
> done

# 8. Alias for generating configs and running them (the old fp-grid behavior)
fp-grid: fp-grid-config fp-grid-run

# 9. Aggregate results
fp-aggregate:
> $(PYTHON) -m experiments.fixed_points.scripts.aggregate --root $(OUT_ROOT) --out $(OUT_ROOT)/aggregate.csv
> @echo "Aggregate → $(OUT_ROOT)/aggregate.csv"

# 10. Clean up
fp-clean:
> rm -rf $(OUT_ROOT)