#!/bin/bash
#SBATCH -J wiki_stance_detection
#SBATCH -o logs/wiki_stance_detection_%A_%a.out
#SBATCH -e logs/wiki_stance_detection_%A_%a.err
#SBATCH -p gpu_requeue
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --account=murphy_lab
#SBATCH --array=0-6  # Adjusted based on the total number of valid experiments

# Change to your project directory and load modules
cd ~/Downloads/thesis
source ../topicexpan/myenv/bin/activate
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd vast

# Define possible parameter values
MODELS=("mlp" "wiki-single" "wiki-dual")
WIKI_MODES=("none" "summary" "content")

# Generate valid combinations (excluding "linear", and restricting "mlp" to "none" only)
EXPERIMENTS=()
for model in "${MODELS[@]}"; do
    if [[ "$model" == "mlp" ]]; then
        EXPERIMENTS+=("$model none")  # Run MLP only with "none"
    else
        for wiki_mode in "${WIKI_MODES[@]}"; do
            EXPERIMENTS+=("$model $wiki_mode")  # Run all wiki modes for "wiki-single" and "wiki-dual"
        done
    fi
done

# Get total number of valid experiments
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

# Select experiment based on SLURM_ARRAY_TASK_ID
EXPERIMENT_ID=$SLURM_ARRAY_TASK_ID
EXPERIMENT="${EXPERIMENTS[$EXPERIMENT_ID]}"
MODEL=$(echo $EXPERIMENT | cut -d' ' -f1)
WIKI_MODE=$(echo $EXPERIMENT | cut -d' ' -f2)

# Run experiment with selected parameters
echo "Running experiment with MODEL=$MODEL, WIKI_MODE=$WIKI_MODE"

python wiki_stance_detection.py --model $MODEL --wiki_mode $WIKI_MODE
