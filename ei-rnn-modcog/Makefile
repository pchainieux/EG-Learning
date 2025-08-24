# =========================
# Makefile (robust & simple)
# =========================
SHELL := /bin/bash
export PYTHONPATH := $(PWD)

# ---- config (override: make all EXP=myexp PIs="0.2 0.5") ----
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

# Resolve scripts dir: supports 'fixed-points/' or 'fixed_points/'
FP_DIR := $(shell if [ -d fixed-points ]; then echo fixed-points; elif [ -d fixed_points ]; then echo fixed_points; else echo fixed-points; fi)

# ---- pretty banner ----
define banner
	@printf "\n\033[1;36m==> %s\033[0m\n" "$(1)"
endef

.PHONY: all deps train fixedpoints analyze robustness aggregate plots ringplots list runs clean

# =========================
# Full pipeline
# =========================
all: deps train fixedpoints analyze robustness aggregate plots

# Minimal deps (avoid heredoc)
deps:
	$(call banner,Checking Python deps)
	@$(PY) -c "import importlib, subprocess, sys; req=['loguru']; \
missing=[r for r in req if importlib.util.find_spec(r) is None]; \
subprocess.check_call([sys.executable,'-m','pip','install','-q']+missing) if missing else print('OK')"

# =========================
# Stage: training (writes seed=000)
# =========================
train:
	$(call banner,TRAIN → $(RUN_ROOT)/$(EXP))
	@set -euo pipefail; \
	for o in $(OPTIMS); do \
	  for p in $(PIs); do \
	    for s in $(SEEDS); do \
	      seedp=$$(printf "%03d" $$s); \
	      outdir="$(RUN_ROOT)/$(EXP)/$$o/pI=$$p/seed=$$seedp"; \
	      mkdir -p "$$outdir"; \
	      echo "[train] $$outdir"; \
	      $(PY) "$(FP_DIR)/train.py" \
	        --optim $$o \
	        --pI $$p \
	        --seed $$s \
	        --task $(TASK) \
	        --hidden $(HIDDEN) \
	        --epochs $(EPOCHS) \
	        --steps $(STEPS) \
	        --seq_len $(FP_SEQ_LEN) \
	        --batch $(FP_BATCH) \
	        --lr_eg $(LR_EG) \
	        --lr_gd $(LR_GD) \
	        --device $(DEVICE) \
	        --outdir "$$outdir"; \
	    done; \
	  done; \
	done


# Helper: iterate over existing runs (robust to seed formatting)
define _for_each_run
set -euo pipefail; \
runs=$$(find "$(RUN_ROOT)/$(EXP)" -type d -name 'seed=*' | sort); \
if [ -z "$$runs" ]; then echo "No runs found under $(RUN_ROOT)/$(EXP)"; exit 0; fi; \
for run in $$runs; do \
  echo "$(1) $$run"; \
  $(2); \
done
endef

# =========================
# Stage: fixed points
# =========================
fixedpoints:
	$(call banner,FIXED POINTS)
	@$(call _for_each_run,[fp], \
	$(PY) "$(FP_DIR)/fixed_point_finder.py" \
	  --run "$$run" \
	  --contexts $(CTX) \
	  --n_init $(FP_N_INIT) \
	  --tol $(FP_TOL) \
	  --max_iter $(FP_MAX_IT) \
	  --seq_len $(FP_SEQ_LEN) \
	  --batch $(FP_BATCH) \
	  --task $(TASK) \
	  --device cpu)

# =========================
# Stage: motif analysis
# =========================
analyze:
	$(call banner,MOTIF ANALYSIS)
	@$(call _for_each_run,[analyze], \
	$(PY) "$(FP_DIR)/analyze_motifs.py" \
	  --run "$$run" \
	  --contexts $(CTX) \
	  --k_slow $(K_SLOW) \
	  --m_overlap $(M_OVERLAP))

# =========================
# Stage: robustness tests
# =========================
robustness:
	$(call banner,ROBUSTNESS)
	@$(call _for_each_run,[robust], \
	$(PY) "$(FP_DIR)/robustness_tests.py" \
	  --run "$$run" \
	  --task $(TASK) \
	  --alphaI $(ALPHA_I) \
	  --noise_std $(NOISE_STD) \
	  --seq_len $(ROB_SEQ_LEN) \
	  --batch $(ROB_BATCH) \
	  --device cpu)

# =========================
# Stage: aggregate + plots
# =========================
aggregate:
	$(call banner,AGGREGATE → $(SUMMARY_CSV))
	@mkdir -p "$(dir $(SUMMARY_CSV))"
	@$(PY) "$(FP_DIR)/aggregate_results.py" --root "$(RUN_ROOT)/$(EXP)" --out "$(SUMMARY_CSV)"

plots:
	$(call banner,PLOTS → $(FIGS_DIR))
	@mkdir -p "$(FIGS_DIR)"
	@$(PY) "$(FP_DIR)/plot_reports.py" --summary "$(SUMMARY_CSV)" --outdir "$(FIGS_DIR)"

# =========================
# Extra: ring fixed-point panels
# =========================
ringplots:
	$(call banner,RING FIXED-POINT PANELS)
	@$(call _for_each_run,[ringplot], \
	$(PY) "$(FP_DIR)/plot_ring_fixed_points.py" \
	  --run "$$run" \
	  --contexts $(CTX) \
	  --seq_len $(FP_SEQ_LEN) \
	  --batch $(FP_BATCH) \
	  --task $(TASK) \
	  --fixation_channel 0 \
	  --mask_threshold 0.5 \
	  --z_mode logit)

# =========================
# Utilities
# =========================
list:
	@printf "Runs under %s:\n" "$(RUN_ROOT)/$(EXP)"; \
	find "$(RUN_ROOT)/$(EXP)" -type d -name 'seed=*' | sort || true

runs: list

clean:
	$(call banner,CLEAN $(RUN_ROOT)/$(EXP))
	@rm -rf "$(RUN_ROOT)/$(EXP)"
