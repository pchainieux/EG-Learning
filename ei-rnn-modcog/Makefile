SHELL := /bin/bash
export PYTHONPATH := $(PWD)

PY ?= python -u
PYTHON ?= $(PY) 

export PY="python3 -u"
export PYTHON="$PY"

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

# =========================================================
# FIXED-POINTS PIPELINE 
# =========================================================

FP_RUN_ROOT    ?= outputs/runs
FP_EXP         ?= ei_pI

OPTIMS         ?= eg gd
PIs            ?= 0.20 0.35 0.50 0.65
SEEDS          ?= 0 1 2 3 4

TASK           ?= dm1
HIDDEN         ?= 256
EPOCHS         ?= 10
STEPS          ?= 500
LR_EG          ?= 1.5
LR_GD          ?= 0.01
DEVICE         ?= auto

CTX            ?= memory anti
FP_N_INIT      ?= 512
FP_TOL         ?= 1e-6
FP_MAX_IT      ?= 800
FP_SEQ_LEN     ?= 350
FP_BATCH       ?= 512

K_SLOW         ?= 4
M_OVERLAP      ?= 3

ALPHA_I        ?= 0.5 0.75 1.0 1.25 1.5
NOISE_STD      ?= 0.0 0.1 0.2
ROB_SEQ_LEN    ?= 350
ROB_BATCH      ?= 128

SUMMARY_CSV    := $(FP_RUN_ROOT)/$(FP_EXP)/summary.csv
FIGS_DIR       := $(FP_RUN_ROOT)/$(FP_EXP)/figs

FP_DIR := experiments/fixed_points

define banner
	@printf "\n\033[1;36m==> %s\033[0m\n" "$(1)"
endef

.PHONY: fp-all fp-deps fp-train fp-fixedpoints fp-analyze fp-robustness fp-aggregate fp-plots fp-ringplots fp-list fp-runs fp-clean

fp-all: fp-deps fp-train fp-fixedpoints fp-analyze fp-robustness fp-aggregate fp-plots

fp-deps:
	$(call banner,Checking Python deps)
	@$(PY) -c "import importlib, subprocess, sys; req=['loguru']; \
missing=[r for r in req if importlib.util.find_spec(r) is None]; \
subprocess.check_call([sys.executable,'-m','pip','install','-q']+missing) if missing else print('OK')"

fp-train:
	$(call banner,TRAIN → $(FP_RUN_ROOT)/$(FP_EXP))
	@set -euo pipefail; \
	for o in $(OPTIMS); do \
	  for p in $(PIs); do \
	    for s in $(SEEDS); do \
	      seedp=$$(printf "%03d" $$s); \
	      outdir="$(FP_RUN_ROOT)/$(FP_EXP)/$$o/pI=$$p/seed=$$seedp"; \
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

define _for_each_run
set -euo pipefail; \
runs=$$(find "$(FP_RUN_ROOT)/$(FP_EXP)" -type d -name 'seed=*' | sort); \
if [ -z "$$runs" ]; then echo "No runs found under $(FP_RUN_ROOT)/$(FP_EXP)"; exit 0; fi; \
for run in $$runs; do \
  echo "$(1) $$run"; \
  $(2); \
done
endef

fp-fixedpoints:
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

fp-analyze:
	$(call banner,MOTIF ANALYSIS)
	@$(call _for_each_run,[analyze], \
	$(PY) "$(FP_DIR)/analyze_motifs.py" \
	  --run "$$run" \
	  --contexts $(CTX) \
	  --k_slow $(K_SLOW) \
	  --m_overlap $(M_OVERLAP))

fp-robustness:
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

fp-aggregate:
	$(call banner,AGGREGATE → $(SUMMARY_CSV))
	@mkdir -p "$(dir $(SUMMARY_CSV))"
	@$(PY) "$(FP_DIR)/aggregate_results.py" --root "$(FP_RUN_ROOT)/$(FP_EXP)" --out "$(SUMMARY_CSV)"

fp-plots:
	$(call banner,PLOTS → $(FIGS_DIR))
	@mkdir -p "$(FIGS_DIR)"
	@$(PY) "$(FP_DIR)/plot_reports.py" --summary "$(SUMMARY_CSV)" --outdir "$(FIGS_DIR)"

fp-ringplots:
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

fp-list:
	@printf "Runs under %s:\n" "$(FP_RUN_ROOT)/$(FP_EXP)"; \
	find "$(FP_RUN_ROOT)/$(FP_EXP)" -type d -name 'seed=*' | sort || true

fp-runs: fp-list

fp-clean:
	$(call banner,CLEAN $(FP_RUN_ROOT)/$(FP_EXP))
	@rm -rf "$(FP_RUN_ROOT)/$(FP_EXP)"

.PHONY: all help
all: eigrad-all fp-all

help:
	@echo "Targets:"
	@echo "  EI gradient:  eigrad-collect eigrad-metrics eigrad-plots eigrad-all [CFG=...]"
	@echo "  Fixed points: fp-deps fp-train fp-fixedpoints fp-analyze fp-robustness fp-aggregate fp-plots fp-ringplots fp-list fp-clean fp-all"
	@echo "Variables:"
	@echo "  EI gradient:  CFG (default: configs/ei_gradient/credit.yaml)"
	@echo "  Fixed points: FP_EXP, FP_RUN_ROOT, OPTIMS, PIs, SEEDS, TASK, HIDDEN, EPOCHS, STEPS, LR_EG, LR_GD, DEVICE,"
	@echo "                CTX, FP_N_INIT, FP_TOL, FP_MAX_IT, FP_SEQ_LEN, FP_BATCH, K_SLOW, M_OVERLAP,"
	@echo "                ALPHA_I, NOISE_STD, ROB_SEQ_LEN, ROB_BATCH"
