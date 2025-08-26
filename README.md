# The Role of Biological Constraints in Shaping Learned Neuronal Representations of Cognitive Tasks

This repository contains the accompanying code for my MSc Machine Learning thesis project. The work investigates biologically plausible learning algorithms by studying the interaction between Exponentiated Gradient Descent (EG) and Dale’s Law, a fundamental biological constraint separating excitatory and inhibitory neurons. The project is structured into two main components:  
- **Analytical results (`analytical_results/`)** – closed-form solutions and theoretical analysis of EG learning dynamics in linear networks.  
- **Empirical results (`ei-rnn-modcog/`)** – large-scale simulations of excitatory/inhibitory recurrent networks trained on the Mod-Cog cognitive benchmark, exploring the representational consequences of biological constraints.  

For full details, please refer to the thesis report as well as the dedicated `README.md` files in each folder.  

## Getting Started

The repository can be set up and executed as follows:

### 1. Clone the repository
```bash
git clone https://github.com/pchainieux/EG-Learning.git
cd EG-Learning
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Each subdirectory provides a dedicated `README.md` with detailed instructions to run their respective expeirments. 