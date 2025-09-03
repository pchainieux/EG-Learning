SHELL := /bin/bash
export PYTHONPATH := $(PWD)

PY ?= python -u
PYTHON ?= $(PY)

# =========================================================
# EI GRADIENT EXPERIMENT
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
# SIMPLE FIXED-POINTS PIPELINE
# ================================

FP_CFG        ?= configs/fixed_points/fixed_points.yaml

FP_RUN_ROOT   ?= outputs/fp_runs
FP_EXP        ?= inh_sweep
INH           ?= 0.10 0.20 0.30

SEED          ?= 1
EPOCHS        ?= 5
STEPS         ?= 200
ONE_FP_SEED   ?= 1   # 1 → pass --one-fp-seed

EXP_DIR       := $(FP_RUN_ROOT)/$(FP_EXP)
SUMMARY_CSV   := $(EXP_DIR)/summary_all.csv

define banner
	@printf "\n\033[1;36m==> %s\033[0m\n" "$(1)"
endef

.PHONY: fp-all fp-sweep fp-aggregate fp-list fp-clean

fp-all: fp-sweep fp-aggregate fp-plots

# Train + analyze + plot per inhibitory fraction
fp-sweep:
	$(call banner,SWEEP inhibitory fractions → $(EXP_DIR))
	@mkdir -p "$(EXP_DIR)"
	@set -euo pipefail; \
	for inh in $(INH); do \
	  outdir="$(EXP_DIR)/inh_$${inh}"; \
	  echo "[run] inh=$$inh  outdir=$$outdir"; \
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

# Concatenate all per-run summary.csv into one CSV (calls a tiny Python module)
fp-aggregate:
	$(call banner,AGGREGATE → $(SUMMARY_CSV))
	@mkdir -p "$(dir $(SUMMARY_CSV))"
	@$(PYTHON) -m experiments.fixed_points.scripts.aggregate --root "$(EXP_DIR)" --out "$(SUMMARY_CSV)"

fp-list:
	@echo "Runs under $(EXP_DIR):"
	@find "$(EXP_DIR)" -type f -name ckpt.pt -printf '%h\n' | sort || true

fp-clean:
	$(call banner,CLEAN $(EXP_DIR))
	@rm -rf "$(EXP_DIR)"

.PHONY: fp-plots
fp-plots:
	$(call banner,PLOTS per run)
	@set -euo pipefail; \
	hits=$$(find "$(FP_RUN_ROOT)/$(FP_EXP)" -type f -name ckpt.pt 2>/dev/null | sort); \
	if [ -z "$$hits" ]; then echo "No runs found under $(FP_RUN_ROOT)/$(FP_EXP)"; exit 0; fi; \
	for ck in $$hits; do \
	  run=$$(dirname "$$ck"); \
	  echo "[plot] $$run"; \
	  $(PYTHON) -m experiments.fixed_points.scripts.plot_fixed_points --run "$$run"; \
	done

help:
	@echo "Targets:"
	@echo "  EI gradient:  eigrad-collect eigrad-metrics eigrad-plots eigrad-all [CFG=...]"
	@echo "  Fixed points: fp-deps fp-train fp-fixedpoints fp-analyze fp-robustness fp-aggregate fp-plots fp-ringplots fp-list fp-clean fp-all"
	@echo "Variables:"
	@echo "  EI gradient:  CFG (default: configs/ei_gradient/credit.yaml)"
	@echo "  Fixed points: FP_EXP, FP_RUN_ROOT, OPTIMS, PIs, SEEDS, TASK, HIDDEN, EPOCHS, STEPS, LR_EG, LR_GD, DEVICE,"
	@echo "                CTX, FP_N_INIT, FP_TOL, FP_MAX_IT, FP_SEQ_LEN, FP_BATCH, K_SLOW, M_OVERLAP,"
	@echo "                ALPHA_I, NOISE_STD, ROB_SEQ_LEN, ROB_BATCH"
