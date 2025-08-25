# experiments/credit_assignment/Makefile

# Usage examples:
#   make collect CFG=configs/credit_default.yaml
#   make metrics CFG=configs/credit_default.yaml
#   make plots   CFG=configs/credit_default.yaml
#   make input-opt CFG=configs/input_opt_default.yaml

PYTHON ?= python
CFG ?= configs/credit_default.yaml

.PHONY: collect metrics plots input-opt clean

collect:
	$(PYTHON) -m scripts.collect_grads --cfg $(CFG)

metrics:
	$(PYTHON) -m scripts.compute_metrics --cfg $(CFG)

plots:
	$(PYTHON) -m scripts.plot_all --cfg $(CFG)

input-opt:
	$(PYTHON) -m scripts.optimize_inputs --cfg $(CFG)

clean:
	rm -rf outputs/*/*.png outputs/*/*.pdf outputs/*/figs outputs/*/*.csv outputs/*/*.json outputs/*/*.pt
