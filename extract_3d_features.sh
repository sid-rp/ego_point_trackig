#!/bin/bash
#SBATCH --job-name=extract_2D
#SBATCH --output=slogs/%A_%a.out  # Log for each job array task
#SBATCH --error=slogs/%A_%a.err   # Error log for each job array task
#SBATCH --time=48:00:00           # Max runtime: 48 hours
#SBATCH --mem=50G                 # 50GB memory per job
#SBATCH --cpus-per-task=4         # 4 CPU cores per task
#SBATCH --gres=gpu:v100:1         # Request 1 V100 GPU per task
#SBATCH --array=0-9               # 10 tasks (100 videos / 10 videos per task)

# Load necessary modules
# module load anaconda  # Adjust based on your system
source activate OSNOM  # Activate the required environment

# Define the list of videos
VIDEOS=("P24_05" "P03_04" "P01_14" "P30_107" "P05_08" "P12_101" "P28_103" "P10_04" "P30_05" 
        "P06_101" "P04_05" "P06_103" "P35_109" "P37_103" "P04_11" "P04_21" "P04_109" "P02_07" 
        "P28_14" "P22_01" "P15_02" "P04_26" "P01_09" "P02_109" "P02_101" "P24_08" "P23_05" 
        "P28_110" "P20_03" "P11_105" "P08_09" "P22_07" "P03_113" "P04_02" "P25_107" "P02_130" 
        "P08_16" "P30_101" "P18_07" "P01_103" "P01_05" "P03_03" "P11_102" "P06_107" "P03_24" 
        "P37_101" "P06_12" "P02_107" "P03_17" "P01_104" "P11_16" "P06_13" "P02_122" "P06_11" 
        "P28_109" "P03_101" "P02_124" "P03_05" "P04_114" "P28_06" "P03_123" "P02_121" "P27_101" 
        "P03_13" "P06_07" "P26_110" "P03_112" "P30_112" "P04_33" "P02_135" "P02_03" "P04_101" 
        "P12_02" "P02_102" "P05_01" "P01_03" "P22_117" "P17_01" "P06_09" "P03_11" "P28_101" 
        "P06_110" "P04_04" "P28_13" "P30_111" "P18_06" "P28_113" "P03_23" "P11_101" "P32_01" 
        "P04_121" "P04_110" "P12_03" "P04_25" "P08_21" "P02_128" "P04_03" "P14_05" "P23_02" 
        "P28_112" "P06_01")

# Define paths
BASE_PATH="/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/"
CODE_PATH="/scratch/projects/fouheylab/dma9300/OSNOM/"
OUTPUT_DIR="osnom_3d_features"

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
