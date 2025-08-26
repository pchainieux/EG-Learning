PY ?= python

CONFIGS_DIR        ?= configs
COUPLING_ONE_CFG   ?= $(CONFIGS_DIR)/coupling_one_layer_config.yaml
COUPLING_TWO_CFG   ?= $(CONFIGS_DIR)/coupling_two_layer_config.yaml
BASIS_ALIGN_CFG    ?= $(CONFIGS_DIR)/basis_alignment_config.yaml
SHALLOW_CFG        ?= $(CONFIGS_DIR)/shallow_config.yaml
TWO_LAYER_CFG      ?= $(CONFIGS_DIR)/two_layer_config.yaml
ONE_MODE_CFG       ?= $(CONFIGS_DIR)/one_mode_case_config.yaml

.PHONY: all
all: coupling-one coupling-two basis-alignment shallow two-layer one-mode-case

.PHONY: coupling-one
coupling-one:
	@echo ">>> Running mode_coupling_one_layer..."
	$(PY) -m scripts.mode_coupling_one_layer --config $(COUPLING_ONE_CFG)

.PHONY: coupling-two
coupling-two:
	@echo ">>> Running mode_coupling_two_layers..."
	$(PY) -m scripts.mode_coupling_two_layers --config $(COUPLING_TWO_CFG)

.PHONY: basis-alignment
basis-alignment:
	@echo ">>> Running basis_alignment..."
	$(PY) -m scripts.basis_alignment --config $(BASIS_ALIGN_CFG)

.PHONY: shallow
shallow:
	@echo ">>> Running one_layer (shallow)..."
	$(PY) -m scripts.one_layer --config $(SHALLOW_CFG)

.PHONY: two-layer
two-layer:
	@echo ">>> Running two_layer..."
	$(PY) -m scripts.two_layer --config $(TWO_LAYER_CFG)

.PHONY: one-mode-case
one-mode-case:
	@echo ">>> Running one_mode_case..."
	$(PY) -m scripts.one_mode_case --config $(ONE_MODE_CFG)

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make all              # run every experiment"
	@echo "  make coupling-one     # one-layer mode coupling"
	@echo "  make coupling-two     # two-layer mode coupling"
	@echo "  make basis-alignment  # basis alignment figure"
	@echo "  make shallow          # shallow (one_layer) run"
	@echo "  make two-layer        # two-layer run"
	@echo "  make one-mode-case    # single-mode case"
	@echo ""
	@echo "Variables you can override, e.g.:"
	@echo "  make PY=python3"
	@echo "  make COUPLING_ONE_CFG=configs/custom_one.yaml"

