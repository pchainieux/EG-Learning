# ==========
# Settings (override on CLI: make all EXP=myexp PIs="0.2 0.4")
# ==========
PY          ?= python
RUN_ROOT    ?= experiments/runs
EXP         ?= ei_pI

# Sweep grid
OPTIMS      ?= eg gd
PIs         ?= 0.20 0.35 0.50 0.65
SEEDS       ?= 0 1 2 3 4

# Training knobs
TASK        ?= dm1
HIDDEN      ?= 256
EPOCHS      ?= 20
STEPS       ?= 1000
LR_EG       ?= 1.5
LR_GD       ?= 0.01
DEVICE      ?= auto

# Fixed-point finder knobs
CTX         ?= memory anti
FP_N_INIT   ?= 512
FP_TOL      ?= 1e-6
FP_MAX_IT   ?= 800
FP_SEQ_LEN  ?= 350
FP_BATCH    ?= 512
FP_ACT      ?= softplus   # match your model’s nonlinearity

# Motif analysis knobs
K_SLOW      ?= 4
M_OVERLAP   ?= 3

# Robustness knobs
ALPHA_I     ?= 0.5 0.75 1.0 1.25 1.5
NOISE_STD   ?= 0.0 0.1 0.2
ROB_SEQ_LEN ?= 350
ROB_BATCH   ?= 128

# ==========
# Derived
# ==========
RUNS := $(foreach o,$(OPTIMS),$(foreach p,$(PIs),$(foreach s,$(SEEDS),$(RUN_ROOT)/$(EXP)/$(o)/pI=$(p)/seed=$(s))))

SUMMARY_CSV := $(RUN_ROOT)/$(EXP)/summary.csv
FIGS_DIR    := $(RUN_ROOT)/$(EXP)/figs

# Make nice logging
define banner
	@printf "\n\033[1;36m==> %s\033[0m\n" $(1)
endef

.PHONY: all train fixedpoints analyze robustness aggregate plots clean runs list

# Full pipeline
all: train fixedpoints analyze robustness aggregate plots

# ==========
# Stage: training
# ==========
train:
	$(call banner,TRAIN: $(words $(RUNS)) runs into $(RUN_ROOT)/$(EXP))
	@set -e; \
	for o in $(OPTIMS); do \
	  for p in $(PIs); do \
	    for s in $(SEEDS); do \
	      outdir="$(RUN_ROOT)/$(EXP)/$$o/pI=$$p/seed=$$s"; \
	      mkdir -p "$$outdir"; \
	      echo "[train] $$outdir"; \
	      $(PY) train.py \
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
	        --outroot $(RUN_ROOT)/$(EXP); \
	    done; \
	  done; \
	done

# ==========
# Stage: fixed points
# ==========
fixedpoints:
	$(call banner,FIXED POINTS over $(words $(RUNS)) runs)
	@set -e; \
	for run in $(RUNS); do \
	  echo "[fp] $$run"; \
	  for ctx in $(CTX); do \
	    $(PY) fixed_point_finder.py \
	      --run "$$run" \
	      --contexts $$ctx \
	      --n_init $(FP_N_INIT) \
	      --tol $(FP_TOL) \
	      --max_iter $(FP_MAX_IT) \
	      --seq_len $(FP_SEQ_LEN) \
	      --batch $(FP_BATCH) \
	      --task $(TASK) \
	      --device cpu; \
	  done; \
	done

# ==========
# Stage: motif reuse / stability analysis
# ==========
analyze:
	$(call banner,MOTIF ANALYSIS over $(words $(RUNS)) runs)
	@set -e; \
	for run in $(RUNS); do \
	  echo "[analyze] $$run"; \
	  $(PY) analyze_motifs.py \
	    --run "$$run" \
	    --contexts $(CTX) \
	    --k_slow $(K_SLOW) \
	    --m_overlap $(M_OVERLAP); \
	done

# ==========
# Stage: robustness tests
# ==========
robustness:
	$(call banner,ROBUSTNESS over $(words $(RUNS)) runs)
	@set -e; \
	for run in $(RUNS); do \
	  echo "[robust] $$run"; \
	  $(PY) robustness_tests.py \
	    --run "$$run" \
	    --task $(TASK) \
	    --alphaI $(ALPHA_I) \
	    --noise_std $(NOISE_STD) \
	    --seq_len $(ROB_SEQ_LEN) \
	    --batch $(ROB_BATCH) \
	    --device cpu; \
	done

# ==========
# Stage: aggregate + plots
# ==========
aggregate:
	$(call banner,AGGREGATE -> $(SUMMARY_CSV))
	@mkdir -p $(dir $(SUMMARY_CSV))
	$(PY) aggregate_results.py --root $(RUN_ROOT)/$(EXP) --out $(SUMMARY_CSV)

plots:
	$(call banner,PLOTS -> $(FIGS_DIR))
	@mkdir -p $(FIGS_DIR)
	$(PY) plot_reports.py --summary $(SUMMARY_CSV) --outdir $(FIGS_DIR)


# ==========
# Fixed Points
# ==========

.PHONY: ringplots
ringplots:
	$(call banner,RING FIXED-POINT PLOTS over $(words $(RUNS)) runs)
	@set -e; \
	for run in $(RUNS); do \
	  echo "[ringplot] $$run"; \
	  PYTHONPATH=$$(pwd) $(PY) -m fixed_points.plot_ring_fixed_points \
	    --run "$$run" \
	    --contexts $(CTX) \
	    --seq_len $(FP_SEQ_LEN) \
	    --batch $(FP_BATCH) \
	    --task $(TASK) \
	    --fixation_channel 0 \
	    --mask_threshold 0.5 \
	    --z_mode logit; \
	done


# ==========
# Utilities
# ==========
list:
	@printf "Runs:\n"; \
	for r in $(RUNS); do echo "  $$r"; done

runs: list

clean:
	$(call banner,CLEAN $(RUN_ROOT)/$(EXP))
	rm -rf $(RUN_ROOT)/$(EXP)
