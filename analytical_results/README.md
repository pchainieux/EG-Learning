# EG Learning: Analytical Study of Exponentiated Gradient Descent

This folder contains the supporting code for the analytical study of the MSc thesis "The role of biological constraints in shaping learned neuronal representations of cognitive tasks", investigating Exponentiated Gradient Descent (EG) as a biologically plausible alternative to standard Gradient Descent (GD).
All graphs and experiments in the report can be reproduced with the code presented here. To run each experiment, change directory to this folder and run the commands outined below. 

## Core analysis scripts

`two_layer.py` - Main experiment comparing EG vs GD dynamics in two layer linear networks. Tracks singular value evolution over training, supports seed averaging for statistical analysis, and demonstrates how EG breaks the modewise independence seen in GD.
```
make two-layer
```

`one_layer.py` - Analyaes shallow linear networks where EG acts element wise on weights. Shows how EG dynamics follow the equation shown in the report, breaking SVD decoupling and creating cross mode interactions that don't exist in GD.
```
make shallow
```

`one_mode_case.py` - Studies the simplest case where networks reduce to scalar chains. Compares analytical ODE solutions with discrete simulations, revealing fundamental differences in convergence scaling.
```
make one-mode-case
```

`mode_coupling_two_layers.py` - Quantifies cross mode interactions in deep networks through rotation matrix analysis. Computes coupling strength and generates animations showing how EG induces continuous basis rotation absent in GD.
```
make coupling-two
```
`mode_coupling_one_layer.py` - Similar mode coupling analysis for shallow networks. Demonstrates that even without depth, EG creates off diagonal interactions in the rotation matrix, fundamentally altering learning dynamics compared to GD's fixed basis evolution.
```
make coupling-one
```
`basis_alignment.py` - Measures how learned singular vectors drift relative to the teacher basis during training. Uses Hungarian algorithm for optimal mode assignment and tracks cosine similarities, quantifying EG's continuous basis drift vs GD's stable alignmenr.
```
make basis-alignment
```

## Folder Structure

```
analytical_results/
├── configs/   
│   ├── basis_alignment_config.yaml
│   ├── coupling_one_layer_config.yaml
│   ├── coupling_two_layer_config.yaml
│   ├── one_mode_case_config.yaml
│   ├── shallow_config.yaml
│   └── two_layer_config.yaml
├── scripts/   
│   ├── basis_alignment.py
│   ├── mode_coupling_one_layer.py
│   ├── mode_coupling_two_layers.py
│   ├── one_layer.py
│   ├── one_mode_case.py
│   └── two_layer.py
└── src/     
    ├── data/
    │   └── data_processing.py 
    ├── inits/
    │   ├── init_shallow.py    
    │   └── init_two_layer.py   
    ├── optim/
    │   ├── eg_sgd_shallow.py   
    │   └── eg_sgd_two_layer.py  
    └── utils/
        ├── vis_style.py      
        └── __init__.py

```
