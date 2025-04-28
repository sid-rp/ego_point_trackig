#!/bin/bash
# Default values (can be overridden by command-line arguments)
GPU_TYPE="RTX8000"  # Default GPU type for NYU
NUM_GPUS=1       # Default number of GPUs

# Parse command-line arguments
while getopts g:n: flag; do
    case "${flag}" in
        g) GPU_TYPE=${OPTARG};;
        n) NUM_GPUS=${OPTARG};;
    esac
done

# Validate the number of GPUs
if ! [[ "$NUM_GPUS" =~ ^[1-8]$ ]]; then
    echo "Error: Number of GPUs must be between 1 and 8."
    exit 1
fi

# Determine the hostname
HOSTNAME=$(hostname)

if [[ $HOSTNAME == *.arc-ts.umich.edu || $HOSTNAME == *.eecs.umich.edu ]]; then
    # Michigan system setup
    GPU_TYPE="A40"
    CPUS=$((16 * NUM_GPUS))
    MEMORY=$((11915 * NUM_CPUS / 1024))  # Convert memory from MB to GB
    PARTITION="spgpu"
    ACCOUNT="fouhey2"
    PROXY_CMD="source /etc/profile.d/http_proxy.sh &&"
else
    # NYU HPC setup
    PROXY_CMD=""
    case $GPU_TYPE in
      "H100")    MEMORY=$((1400 * NUM_GPUS / 4)); CPUS=$((96 * NUM_GPUS / 4));;
      "A100")    MEMORY=$((480 * NUM_GPUS / 4));  CPUS=$((64 * NUM_GPUS / 4));;
      "V100")    MEMORY=$((360 * NUM_GPUS / 4));  CPUS=$((40 * NUM_GPUS / 4));;
      "RTX8000") MEMORY=$((360 * NUM_GPUS / 4));  CPUS=$((48 * NUM_GPUS / 4));;
      *) echo "Unsupported GPU type: $GPU_TYPE"; exit 1;;
    esac
fi

# Submit the job
sbatch --account=pr_96_general \
       --time=24:00:00 \
       --mem=${MEMORY}G \
       --gres=gpu:${GPU_TYPE,,}:$NUM_GPUS \
       --cpus-per-task=$CPUS \
       ${PARTITION:+--partition=$PARTITION} \
       ${ACCOUNT:+--account=$ACCOUNT} \
       --job-name="trainer_${GPU_TYPE}_${NUM_GPUS}gpus" \
       --output=slogs/%j.out \
       --error=slogs/%j.err \
       --wrap="\
       source /scratch/projects/fouheylab/dma9300/miniconda3/etc/profile.d/conda.sh && \
       conda activate OSNOM && \
       echo 'Using Python:' \$(which python) && \
       cd /scratch/projects/fouheylab/dma9300/OSNOM/ && \
       python trainer.py \
       --batch_size 32 \
       --epochs 70 \
       --learning_rate 0.000001 \
       --num_workers 8 \
       --warmup_epochs 0 \
       --min_lr 1e-6 \
       --arch croco \
       --use_delta True \
       --loss_function combined \
       --output_dir croco_model_epochs_70_combined_loss_deltanet\
       --print_freq 20 \
       --write_freq 50"
