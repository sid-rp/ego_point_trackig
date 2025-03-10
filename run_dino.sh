#!/bin/bash

# Default values (can be overridden by command-line arguments)
NUM_CPUS=4       # Number of CPUs
MEMORY=100        # Memory in GB

# Create slogs directory if it doesn't exist
mkdir -p slogs

# Submit the job with memory in GB
sbatch --account=pr_96_general \
       --time=72:00:00 \
       --mem=${MEMORY}G \
       --cpus-per-task=$NUM_CPUS \
       --job-name="dino_2d_features_CPU" \
       --output=slogs/%j.out \
       --error=slogs/%j.err \
       --wrap="\
       eval \"\$(conda shell.bash hook)\" && \
       conda activate OSNOM && \
       python /scratch/projects/fouheylab/dma9300/OSNOM/code/tracking_code/scripts/extract_feat_2D.py"
