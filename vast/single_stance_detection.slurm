#!/bin/bash
#SBATCH -J single_stance_detection
#SBATCH -o logs/single_stance_%A_%a.out
#SBATCH -e logs/single_stance_%A_%a.err
#SBATCH -p gpu_requeue
#SBATCH -t 4:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --account=murphy_lab
#SBATCH --array=0-5

# Change to your project directory and load modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd vast

# Define arrays for flag combinations.
# There are 3 model types and 2 prefix options.
models=("linear" "attention" "cross")
# For the prefix flag, we decide to use:
#  index 0: --use_prefix (i.e. using the default prefix)
#  index 1: --no_prefix  (i.e. disable the prefix)
prefix_options=("use" "no")

# Compute indices from SLURM_ARRAY_TASK_ID:
# Let the task id index from 0 to 5.
# Model index is TASK_ID mod 3, and prefix index is TASK_ID divided by 3 (integer division).
model_index=$(( SLURM_ARRAY_TASK_ID % 3 ))
prefix_index=$(( SLURM_ARRAY_TASK_ID / 3 ))

model_flag=${models[$model_index]}
if [ "$prefix_index" -eq 0 ]; then
    prefix_flag="--use_prefix"
else
    prefix_flag="--no_prefix"
fi

echo "Running experiment with model: ${model_flag} and prefix flag: ${prefix_flag}"

# Run the Python script with the chosen flags.
python single_stance_detection.py --model ${model_flag} ${prefix_flag}
