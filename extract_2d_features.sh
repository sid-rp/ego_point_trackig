#!/bin/bash
#SBATCH --job-name=extract_2D
#SBATCH --output=slogs/%A_%a.out  # Log for each job array task
#SBATCH --error=slogs/%A_%a.err   # Error log for each job array task
#SBATCH --time=48:00:00           # Max runtime: 48 hours
#SBATCH --mem=50G                 # 50GB memory per job
#SBATCH --cpus-per-task=4         # 4 CPU cores per task
#SBATCH --gres=gpu:1         # Request 1 GPU per task
#SBATCH --constraint="v100|rtx8000"  # Request either V100 or RTX 8000
#SBATCH --array=0-10               # 10 tasks (110 videos / 10 videos per task)

# Load necessary modules
# module load anaconda  # Adjust based on your system
# module load anaconda  # Adjust based on your system
source /scratch/projects/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"  # Ensures Conda works in non-interactive SLURM jobs
conda activate OSNOM


# Define the list of videos, we have a video_ids.txt file which contains video names one per line
VIDEOS_FILE="./video_ids.txt"
if [ ! -f "$VIDEOS_FILE" ]; then
    echo "File $VIDEOS_FILE does not exist!"
    exit 1
fi
mapfile -t VIDEOS < "$VIDEOS_FILE"


# Define paths
BASE_PATH="/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/"
CODE_PATH=$(pwd)
OUTPUT_DIR="osnom_2d_features"

# Compute the start index for this job
TASK_ID=$SLURM_ARRAY_TASK_ID
START_INDEX=$((TASK_ID * 10))
END_INDEX=$((START_INDEX + 10))

# Process 10 videos per GPU
for ((i=START_INDEX; i<END_INDEX && i<${#VIDEOS[@]}; i++)); do
    VIDEO=${VIDEOS[i]}
    echo "Processing video: $VIDEO on GPU ${SLURM_ARRAY_TASK_ID}"

    python $CODE_PATH/code/tracking_code/extract_feat/save_feat_batch_2D.py \
        --output_dir $OUTPUT_DIR \
        --data_path $BASE_PATH/data/aggregated/$VIDEO \
        --frames_path $BASE_PATH/data/preprocessed_data/${VIDEO}.tar \
        --kitchen $VIDEO 
done

# Wait for all background jobs to finish
wait
