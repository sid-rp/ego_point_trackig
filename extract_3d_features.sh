#!/bin/bash
#SBATCH --job-name=extract_3D
#SBATCH --output=slogs/%A_%a.out  # Log for each job array task
#SBATCH --error=slogs/%A_%a.err   # Error log for each job array task
#SBATCH --time=48:00:00           # Max runtime: 48 hours
#SBATCH --mem=50G                 # 50GB memory per job
#SBATCH --cpus-per-task=4         # 4 CPU cores per task
#SBATCH --gres=gpu:1              # Request 1  GPU per task
#SBATCH --array=0-9               # 10 tasks (100 videos / 10 videos per task)

# Load necessary modules
# module load anaconda  # Adjust based on your system
source activate OSNOM  # Activate the required environment

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
OUTPUT_DIR="/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/osnom_3d_features"

# Compute the start index for this job
TASK_ID=$SLURM_ARRAY_TASK_ID
START_INDEX=$((TASK_ID * 10))
END_INDEX=$((START_INDEX + 10))

# Process 10 videos per GPU
for ((i=START_INDEX; i<END_INDEX && i<${#VIDEOS[@]}; i++)); do
    VIDEO=${VIDEOS[i]}
    echo "Processing video: $VIDEO on GPU ${SLURM_ARRAY_TASK_ID}"

    python $CODE_PATH/code/tracking_code/extract_feat/save_feat_batch_3D.py \
        --output_dir $OUTPUT_DIR \
        --data_path $BASE_PATH/data/aggregated/$VIDEO \
        --frames_path $BASE_PATH/data/preprocessed_data/${VIDEO}.tar \
        --mesh_path  $BASE_PATH/data/aggregated/$VIDEO \
        --kitchen $VIDEO 
done

# Wait for all background jobs to finish
wait
