SHELL := /bin/bash
export PYTHONPATH := $(PWD)

PY ?= python -u
PYTHON ?= $(PY)

# =========================================================
# EI GRADIENT EXPERIMENT (as before)
# =========================================================

CFG ?= configs/ei_gradient/credit.yaml

.PHONY: eigrad-collect eigrad-metrics eigrad-plots eigrad-all

eigrad-collect:
	$(PYTHON) -m experiments.ei_gradient.scripts.collect_grads --cfg $(CFG)

eigrad-metrics:
	$(PYTHON) -m experiments.ei_gradient.scripts.compute_metrics --cfg $(CFG)

eigrad-plots:
	$(PYTHON) -m experiments.ei_gradient.scripts.plot_all --cfg $(CFG)

eigrad-all: eigrad-collect eigrad-metrics eigrad-plots


# ================================
# FIXED-POINTS PIPELINE (simple & robust)
# ================================
# Base config for fixed-points runs
FP_CFG        ?= configs/fixed_points/fixed_points.yaml

# Where to store runs
FP_RUN_ROOT   ?= outputs/fp_runs
FP_EXP        ?= inh_sweep

# Per-run controls
NAME          ?= default
OUTDIR        ?= $(FP_RUN_ROOT)/$(FP_EXP)/$(NAME)

# Sweep controls
INH           ?= 0.10 0.20 0.30
SEED          ?= 1
EPOCHS        ?= 5
STEPS         ?= 200
ONE_FP_SEED   ?= 1   # 1 → pass --one-fp-seed to orchestrator

# Aggregate CSV for a sweep
SUMMARY_CSV   ?= $(FP_RUN_ROOT)/$(FP_EXP)/summary_all.csv

.PHONY: help \
        fp-all fp-run fp-train fp-rollout fp-solve fp-plot \
        fp-sweep fp-aggregate fp-plots fp-list fp-clean

help:
	@echo "Targets:"
	@echo "  EI gradient:  eigrad-collect eigrad-metrics eigrad-plots eigrad-all [CFG=...]"
	@echo "  Fixed points:"
	@echo "    fp-run       : End-to-end for a single OUTDIR (train -> seeds+solve -> plots)"
	@echo "    fp-all       : Alias of fp-run"
	@echo "    fp-train     : Train only"
	@echo "    fp-rollout   : Seeds (orchestrator currently also solves)"
	@echo "    fp-solve     : Solve-only (needs OUTDIR/run/ckpt.pt and seeds)"
	@echo "    fp-plot      : Plot-only"
	@echo "    fp-sweep     : Loop over INH values and run end-to-end with plots"
	@echo "    fp-aggregate : Concatenate per-run summaries into one CSV"
	@echo "    fp-plots     : Re-plot for all runs under FP_RUN_ROOT/FP_EXP"
	@echo "    fp-list      : List run directories found under FP_RUN_ROOT/FP_EXP"
	@echo "    fp-clean     : Remove outputs under FP_RUN_ROOT/FP_EXP"
	@echo ""
	@echo "Variables (override as needed):"
	@echo "  FP_CFG=$(FP_CFG)"
	@echo "  FP_RUN_ROOT=$(FP_RUN_ROOT)"
	@echo "  FP_EXP=$(FP_EXP)"
	@echo "  NAME=$(NAME)"
	@echo "  OUTDIR=$(OUTDIR)"
	@echo "  INH=$(INH)"
	@echo "  SEED=$(SEED)"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  STEPS=$(STEPS)"
	@echo "  ONE_FP_SEED=$(ONE_FP_SEED)"
	@echo "  SUMMARY_CSV=$(SUMMARY_CSV)"

# ----------------
# Single run flow
# ----------------

# End-to-end single run (train -> seeds+solve -> plots)
fp-run:
	@echo "[fp-run] OUTDIR=$(OUTDIR)"
	@mkdir -p "$(OUTDIR)"
	$(PYTHON) -m experiments.fixed_points.scripts.run_fixed_points \
		--config $(FP_CFG) \
		--outdir "$(OUTDIR)" \
		--plot

# Alias
fp-all: fp-run

# Train only
fp-train:
	@echo "[fp-train] OUTDIR=$(OUTDIR)"
	@mkdir -p "$(OUTDIR)"
	$(PYTHON) -m base_scripts.train_singlehead_modcog \
		--config $(FP_CFG) \
		--outdir "$(OUTDIR)"

# Rollout seeds (note: current orchestrator will also run solver)
fp-rollout:
	@echo "[fp-rollout] OUTDIR=$(OUTDIR)"
	@mkdir -p "$(OUTDIR)"
	$(PYTHON) -m experiments.fixed_points.scripts.run_fixed_points \
		--config $(FP_CFG) \
		--outdir "$(OUTDIR)"

# Solve-only (requires OUTDIR/run/ckpt.pt and rollout_seeds.npz)
fp-solve:
	@echo "[fp-solve] OUTDIR=$(OUTDIR)"
	$(PYTHON) -m experiments.fixed_points.scripts.unified_fixed_points \
		--run "$(OUTDIR)/run" \
		--config $(FP_CFG)

# Plot-only
fp-plot:
	@echo "[fp-plot] OUTDIR=$(OUTDIR)"
	$(PYTHON) -m experiments.fixed_points.scripts.plot_fixed_points \
		--run "$(OUTDIR)/run" \
		--config $(FP_CFG)

# ----------------
# Simple sweep over inhibition fraction
# ----------------
fp-sweep:
	@echo "[fp-sweep] EXP=$(FP_EXP)  ROOT=$(FP_RUN_ROOT)  INH=$(INH)"
	@mkdir -p "$(FP_RUN_ROOT)/$(FP_EXP)"
	@set -euo pipefail; \
	for inh in $(INH); do \
	  outdir="$(FP_RUN_ROOT)/$(FP_EXP)/inh_$${inh}"; \
	  echo "[fp-sweep] inh=$$inh  outdir=$$outdir  seed=$(SEED) epochs=$(EPOCHS) steps=$(STEPS)"; \
	  extra=""; \
	  if [ "$(ONE_FP_SEED)" = "1" ]; then extra="--one-fp-seed"; fi; \
	  $(PYTHON) -m experiments.fixed_points.scripts.run_fixed_points \
	    --config $(FP_CFG) \
	    --outdir "$$outdir" \
	    --inh $$inh \
	    --seed $(SEED) \
	    --epochs $(EPOCHS) \
	    --steps-per-epoch $(STEPS) \
	    $$extra \
	    --plot; \
	done

# Aggregate all per-run summaries under FP_RUN_ROOT/FP_EXP into one CSV
fp-aggregate:
	@echo "[fp-aggregate] writing $(SUMMARY_CSV)"
	@mkdir -p "$(dir $(SUMMARY_CSV))"
	$(PYTHON) -m experiments.fixed_points.scripts.aggregate \
		--root "$(FP_RUN_ROOT)/$(FP_EXP)" \
		--out  "$(SUMMARY_CSV)"

# Re-plot figures for all runs under FP_RUN_ROOT/FP_EXP
fp-plots:
	@echo "[fp-plots] scanning $(FP_RUN_ROOT)/$(FP_EXP)"
	@set -euo pipefail; \
	hits=$$(find "$(FP_RUN_ROOT)/$(FP_EXP)" -type f -name ckpt.pt 2>/dev/null | sort); \
	if [ -z "$$hits" ]; then echo "No runs found under $(FP_RUN_ROOT)/$(FP_EXP)"; exit 0; fi; \
	for ck in $$hits; do \
	  run=$$(dirname "$$ck"); \
	  echo "[plot] $$run"; \
	  $(PYTHON) -m experiments.fixed_points.scripts.plot_fixed_points \
	    --run "$$run" \
	    --config $(FP_CFG); \
	done

# List runs discovered (by ckpt presence)
fp-list:
	@echo "[fp-list] runs under $(FP_RUN_ROOT)/$(FP_EXP):"
	@find "$(FP_RUN_ROOT)/$(FP_EXP)" -type f -name ckpt.pt -printf '%h\n' 2>/dev/null | sort || true

# Clean outputs for this experiment tag
fp-clean:
	@echo "[fp-clean] removing $(FP_RUN_ROOT)/$(FP_EXP)"
	@rm -rf "$(FP_RUN_ROOT)/$(FP_EXP)"
