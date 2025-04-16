#!/bin/bash
#SBATCH --job-name=cpu_preprocess_job
#SBATCH --time=24:00:00    # Each job handles 5 indices, so total time is ~5 hours
#SBATCH --cpus-per-task=1  # Each task uses 1 CPU
#SBATCH --ntasks=1
#SBATCH --mem=50G         # Allocate 100 GB of memory to each CPU
#SBATCH --array=0-21       # 18 tasks, each handling a specific index range (0 to 17)
#SBATCH --output=slogs/%A_%a.out
#SBATCH --error=slogs/%A_%a.err

# Activate conda environment and run the Python script for the calculated index range
source /scratch/projects/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh
conda activate mast3r
cd /scratch/projects/fouheylab/dma9300/OSNOM/

# Set up the root directory where the .npz files are stored
ROOT="/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/sparse_cameras/"

# Determine the file range for this specific array job task
start_idx=$((SLURM_ARRAY_TASK_ID * 5))       # Calculate the start index
end_idx=$((start_idx + 5))                   # Calculate the end index for this job

# Run the Python script to compute the overlaps
python generate_overlapping_pairs_epic.py --start_idx $start_idx --end_idx $end_idx --root $ROOT
