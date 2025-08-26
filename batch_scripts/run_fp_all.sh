#!/bin/bash
#SBATCH --job-name=fp_all            # Job name shown by squeue
#SBATCH --output=logs/fp_all_%j.out  # Std‑out file (%j will be replaced by job ID)
#SBATCH --error=logs/fp_all_%j.err   # Std‑err file
#SBATCH --mail-user=paul.chainieux.24@ucl.ac.uk   # Email for notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=32:00:00
#SBATCH --partition=gpu              # Use a GPU partition
#SBATCH --gres=gpu:1                 # Request one GPU

echo "Job starting on node: $SLURMD_NODENAME at $(date)"

# ----------------------------------------------------------------------
# Ensure the logs directory exists
mkdir -p logs

# ----------------------------------------------------------------------
# Initialize and activate Conda
# ----------------------------------------------------------------------
# Adjust the path below if your Conda is installed elsewhere
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: could not find conda.sh at $HOME/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

# Activate your environment (replace lr_env with your actual env name)
conda activate lr_env

# ----------------------------------------------------------------------
# Run the fixed‑points experiment
# ----------------------------------------------------------------------
# Change into the ei‑rnn‑modcog directory of your EG‑Learning clone
cd "$HOME/EG-Learning/ei-rnn-modcog" || { echo "Failed to cd into repository"; exit 1; }

# Optionally override Makefile variables here.  For example, to force
# training to use CUDA you can set DEVICE=cuda.  Leave variables unset
# to use the Makefile defaults.
export DEVICE=cuda        # use GPU for training stage
export FP_EXP=ei_pI       # name of the experiment (default: ei_pI)

echo "Running fixed points pipeline in $(pwd)"
make fp-all

# ----------------------------------------------------------------------
# Clean up
# ----------------------------------------------------------------------
conda deactivate
echo "Job completed successfully on $(date)"
