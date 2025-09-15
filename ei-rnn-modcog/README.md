# EI-RNN ModCog: Excitatory-Inhibitory RNNs for Mod-Cog

This folder contains the supporting code for the empirical study of the MSc thesis "The role of biological constraints in shaping learned neuronal representations of cognitive tasks", investigating how Excitatory-Inhibitory Recurrent Neural Networks learn cognitive tasks under biological constraints like Dale's law. All empirical results and network analyses in the thesis can be reproduced with the code presented here.

## Core analysis scripts

`train_singlehead_modcog.py` - Main training script for E/I RNN models on cognitive tasks. Implements Dale's law constraints, supports both EG and GD optimization, and includes spectral radius normalisation. Tracks training metrics and saves checkpoints for downstream analysis.
```
python train_singlehead_modcog.py --config configs/base_training/config.yaml
```

`train_lr_sweep.py` - Systematic learning rate comparison between EG and GD optimisers. Evaluates convergence properties, plateau detection, and generates performance curves across learning rate ranges. 
```
python train_lr_sweep.py --config configs/lr_sweep/lr_sweep.yaml
```

`collect_grads.py` - Gradient attribution analysis tool that hooks into trained E/I networks to collect hidden state gradients during task execution. Separates contributions by excitatory and inhibitory populations, enabling credit assignment analysis.
```
make eigrad-collect CFG=configs/ei_gradient/credit.yaml
```

`compute_metrics.py` - Computes comprehensive gradient based metrics including L2 norms, cosine alignments, and Fisher information across E/I populations. Quantifies how biological constraints affect learning dynamics and credit assignment.
```
make eigrad-metrics CFG=configs/ei_gradient/credit.yaml
```

`plot_all.py` - Generates visualisations of E/I gradient dynamics. Creates timecourse plots, distribution comparisons, and saliency maps showing differential learning between excitatory and inhibitory populations.
```
make eigrad-plots CFG=configs/ei_gradient/credit.yaml
```

`Fixed Point Sweep` - Systematic analysis across inhibition levels to study E/I balance effects.
```
make fp-sweep FP_EXP=inh_analysis INH="0.1 0.2 0.3 0.4"
```

## Folder Structure

```
ei-rnn-modcog/
├── base_scripts/
│   ├── __pycache__/
│   ├── analyze_ckpt.py
│   ├── eval_checkpoint.py
│   └── train_singlehead_modcog.py
├── configs/
│   ├── ablation/
│   │   ├── control.yaml
│   │   ├── leak_0_2.yaml
│   │   ├── leak_0_5.yaml
│   │   ├── leak_1.yaml
│   │   ├── nonlinearity_softplus.yaml
│   │   ├── nonlinearity_tanh.yaml
│   │   ├── readout_all.yaml
│   │   ├── readout_e_only.yaml
│   │   ├── row_sum_0_5.yaml
│   │   └── row_sum_0.yaml
│   ├── base_training/
│   │   └── config.yaml
│   ├── ei_gradient/
│   │   └── credit.yaml
│   ├── fixed_points/
│   │   └── fixed_point.yaml
│   ├── lr_sweep/
│   │   └── lr_sweep.yaml
│   └── model_perf/
│       ├── t_1_H_128.yaml
│       ├── t_1_H_256.yaml
│       ├── t_5_H_128.yaml
│       └── t_5_H_256.yaml
├── experiments/
│   ├── ei_gradient/
│   │   ├── scripts/
│   │   │   ├── __pycache__/
│   │   │   ├── collect_grads.py
│   │   │   ├── compute_metrics.py
│   │   │   └── plot_all.py
│   │   └── src/
│   │       ├── __pycache__/
│   │       ├── __init__.py
│   │       ├── attribution.py
│   │       ├── hooks.py
│   │       ├── io_utils.py
│   │       └── metrics.py
│   ├── fixed_points/
│   │   ├── __pycache__/
│   │   └── scripts/
│   │       ├── __pycache__/
│   │       ├── method_1/
│   │       │   ├── aggregate.py
│   │       │   ├── plot_fixed_points.py
│   │       │   ├── plot_ring_panels.py
│   │       │   ├── run_fixed_points.py
│   │       │   └── unified_fixed_points.py
│   │       ├── method_2/
│   │       │   ├── plot_ring_panels.py
│   │       │   ├── run_fixed_points.py
│   │       │   └── unified_fixed_points.py
│   │       └── plot_trajectories/
│   │           ├── config_pca.yaml
│   │           ├── dlygo_pca.py
│   │           └── __init__.py
│   │   └── src/
│   │       ├── __pycache__/
│   │       ├── config.py
│   │       ├── model_io.py
│   │       └── __init__.py
│   └── lr_sweep/
│       ├── __pycache__/
│       └── train_lr_sweep.py
├── src/
│   ├── analysis/
│   │   ├── __pycache__/
│   │   ├── compare_eg_gd.py
│   │   ├── summary.py
│   │   ├── viz_compare.py
│   │   ├── viz_training.py
│   │   └── viz_weights.py
│   ├── data/
│   │   ├── __pycache__/
│   │   ├── dataset_cached.py
│   │   └── mod_cog_tasks.py
│   ├── models/
│   │   ├── __pycache__/
│   │   └── ei_rnn.py
│   ├── optim/
│   │   ├── __pycache__/
│   │   └── sgd_eg.py
│   ├── tests/
│   │   ├── __pycache__/
│   │   ├── conftest.py
│   │   ├── test_data.py
│   │   ├── test_ei_constraints.py
│   │   ├── test_losses.py
│   │   └── test_spectral.py
│   ├── training/
│   │   ├── __pycache__/
│   │   ├── losses.py
│   │   └── metrics.py
│   └── utils/
│       ├── __pycache__/
│       ├── plot_ablation_graphs.py
│       └── seeding.py
├── neurogym/
├── .gitignore
└── Makefile
```
