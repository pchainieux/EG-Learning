# =========================
# Makefile (robust paths)
# =========================
SHELL := /bin/bash

# ---- knobs (override: make all EXP=myexp PIs="0.2 0.5") ----
PY          ?= python
RUN_ROOT    ?= experiments/runs
EXP         ?= ei_pI

OPTIMS      ?= eg gd
PIs         ?= 0.20 0.35 0.50 0.65
SEEDS       ?= 0 1 2 3 4

TASK        ?= dm1
HIDDEN      ?= 256
EPOCHS      ?= 20
STEPS       ?= 1000
LR_EG       ?= 1.5
LR_GD       ?= 0.01
DEVICE      ?= auto

CTX         ?= memory anti
FP_N_INIT   ?= 512
FP_TOL      ?= 1e-6
FP_MAX_IT   ?= 800
FP_SEQ_LEN  ?= 350
FP_BATCH    ?= 512

K_SLOW      ?= 4
M_OVERLAP   ?= 3

ALPHA_I     ?= 0.5 0.75 1.0 1.25 1.5
NOISE_STD   ?= 0.0 0.1 0.2
ROB_SEQ_LEN ?= 350
ROB_BATCH   ?= 128

SUMMARY_CSV := $(RUN_ROOT)/$(EXP)/summary.csv
FIGS_DIR    := $(RUN_ROOT)/$(EXP)/figs

# ---- pretty banner ----
define banner
	@printf "\n\033[1;36m==> %s\033[0m\n" "$(1)"
endef

# ---- helper: run a script that lives under fixed_points/ (or fixed-points/) ----
# Usage: $(call RUN_FP,train, --arg1 foo --arg2 bar)
define RUN_FP
if [ -d fixed_points ]; then \
	PYTHONPATH=$$(pwd) $(PY) -m fixed_points.$(1) $(2); \
elif [ -d fixed-points ]; then \
	$(PY) fixed-points/$$(echo "$(1)" | tr . /).py $(2); \
else \
	echo "ERROR: Neither fixed_points/ nor fixed-points/ exists."; \
	exit 1; \
fi
endef

# ---- helper: loop over existing run dirs (robust to seed=0 vs seed=000) ----
# Expands at execution time.
define FOR_EACH_RUN
for run in $$(find "$(RUN_ROOT)/$(EXP)" -type d -name 'seed=*' | sort); do \
	echo "$(1) $$run"; \
	$(2); \
done
endef

.PHONY: all train fixedpoints analyze robustness aggregate plots ringplots clean list runs


# =========================
# Dependencied
# =========================

.PHONY: deps

deps:
	$(call banner,Checking Python dependencies)
	@$(PY) - <<'PY'
import importlib, subprocess, sys
pkgs = [
    "loguru",        # neurogym logging
    "numpy", "scipy", "pandas",
    "matplotlib", "scikit-learn"
]
missing = []
for p in pkgs:
    try:
        importlib.import_module(p.replace("-", "_"))
    except Exception:
        missing.append(p)
if missing:
    print("Installing:", " ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
else:
    print("OK: all deps present.")
PY

# =========================
# Full pipeline
# =========================
all: train fixedpoints analyze robustness aggregate plots

# =========================
# Stage: training
# =========================
train:
	$(call banner,TRAIN → writing under $(RUN_ROOT)/$(EXP))
	@set -e; \
	for o in $(OPTIMS); do \
	  for p in $(PIs); do \
	    for s in $(SEEDS); do \
	      seedp=$$(printf "%03d" $$s); \
	      outdir="$(RUN_ROOT)/$(EXP)/$$o/pI=$$p/seed=$$seedp"; \
	      mkdir -p "$$outdir"; \
	      echo "[train] $$outdir"; \
	      $(call RUN_FP,train, \
	        --optim $$o \
	        --pI $$p \
	        --seed $$s \
	        --task $(TASK) \
	        --hidden $(HIDDEN) \
	        --epochs $(EPOCHS) \
	        --steps $(STEPS) \
	        --lr_eg $(LR_EG) \
	        --lr_gd $(LR_GD) \
	        --device $(DEVICE) \
	        --outroot $(RUN_ROOT)/$(EXP) ); \
	    done; \
	  done; \
	done

# =========================
# Stage: fixed points
# =========================
fixedpoints:
	$(call banner,FIXED POINTS over existing runs in $(RUN_ROOT)/$(EXP))
	@set -e; \
	$(call FOR_EACH_RUN,[fp], \
	  $(call RUN_FP,fixed_point_finder, \
	    --run "$$run" \
	    --contexts $(CTX) \
	    --n_init $(FP_N_INIT) \
	    --tol $(FP_TOL) \
	    --max_iter $(FP_MAX_IT) \
	    --seq_len $(FP_SEQ_LEN) \
	    --batch $(FP_BATCH) \
	    --task $(TASK) \
	    --device cpu ))

# =========================
# Stage: motif analysis
# =========================
analyze:
	$(call banner,MOTIF ANALYSIS over existing runs)
	@set -e; \
	$(call FOR_EACH_RUN,[analyze], \
	  $(call RUN_FP,analyze_motifs, \
	    --run "$$run" \
	    --contexts $(CTX) \
	    --k_slow $(K_SLOW) \
	    --m_overlap $(M_OVERLAP) ))

# =========================
# Stage: robustness tests
# =========================
robustness:
	$(call banner,ROBUSTNESS over existing runs)
	@set -e; \
	$(call FOR_EACH_RUN,[robust], \
	  $(call RUN_FP,robustness_tests, \
	    --run "$$run" \
	    --task $(TASK) \
	    --alphaI $(ALPHA_I) \
	    --noise_std $(NOISE_STD) \
	    --seq_len $(ROB_SEQ_LEN) \
	    --batch $(ROB_BATCH) \
	    --device cpu ))

# =========================
# Stage: aggregate + plots
# =========================
aggregate:
	$(call banner,AGGREGATE → $(SUMMARY_CSV))
	@mkdir -p $(dir $(SUMMARY_CSV))
	@$(call RUN_FP,aggregate_results, --root $(RUN_ROOT)/$(EXP) --out $(SUMMARY_CSV))

plots:
	$(call banner,PLOTS → $(FIGS_DIR))
	@mkdir -p $(FIGS_DIR)
	@$(call RUN_FP,plot_reports, --summary $(SUMMARY_CSV) --outdir $(FIGS_DIR))

# =========================
# Extra: ring fixed-point panels
# =========================
ringplots:
	$(call banner,RING FIXED-POINT PLOTS over existing runs)
	@set -e; \
	$(call FOR_EACH_RUN,[ringplot], \
	  $(call RUN_FP,plot_ring_fixed_points, \
	    --run "$$run" \
	    --contexts $(CTX) \
	    --seq_len $(FP_SEQ_LEN) \
	    --batch $(FP_BATCH) \
	    --task $(TASK) \
	    --fixation_channel 0 \
	    --mask_threshold 0.5 \
	    --z_mode logit ))

# =========================
# Utilities
# =========================
list:
	@printf "Runs under %s:\n" "$(RUN_ROOT)/$(EXP)"
	@find "$(RUN_ROOT)/$(EXP)" -type d -name 'seed=*' | sort || true

runs: list

clean:
	$(call banner,CLEAN $(RUN_ROOT)/$(EXP))
	@rm -rf "$(RUN_ROOT)/$(EXP)"
