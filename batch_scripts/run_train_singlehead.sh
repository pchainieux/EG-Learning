#!/bin/bash
#SBATCH --job-name=fp_all
#SBATCH --output=logs/fp_all_%j.out
#SBATCH --error=logs/fp_all_%j.err
#SBATCH --mail-user=paul.chainieux.24@ucl.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=32:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

echo "Job starting on node: $SLURMD_NODENAME at $(date)"
mkdir -p logs

# Conda env
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh 2>/dev/null || true; fi
module load miniconda/23.10.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/modcog"   # <-- your env path

# Sanity checks (optional)
python -V
which python
nvidia-smi || true

# Thread hygiene
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

cd "$HOME/EG-Learning/ei-rnn-modcog" || { echo "Failed to cd into repository"; exit 1; }
export DEVICE=cuda
export FP_EXP=ei_pI

echo "Training ei-rnn pipeline in $(pwd)"
python -u -m base_scripts.train_singlehead_modcog --config configs/base_training/config.yaml

conda deactivate || true
echo "Job completed successfully on $(date)"
