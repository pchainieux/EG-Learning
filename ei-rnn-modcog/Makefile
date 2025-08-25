# Makefile for EI credit assignment experiment

# Default interpreter and configuration file
PYTHON ?= python
CFG ?= ei_gradient/configs/credit.yaml

.PHONY: collect metrics plots clean

# Collect hidden states and gradients
collect:
	$(PYTHON) -m ei_gradient.scripts.collect_grads --cfg $(CFG)

# Compute metrics and summaries
metrics:
	$(PYTHON) -m ei_gradient.scripts.compute_metrics --cfg $(CFG)

# Generate figures from computed metrics
plots:
	$(PYTHON) -m ei_gradient.scripts.plot_all --cfg $(CFG)

# Remove generated outputs (figures, csv, json, tensors)
clean:
	rm -rf outputs/*/*.png outputs/*/*.pdf outputs/*/figs outputs/*/*.csv outputs/*/*.json outputs/*/*.pt