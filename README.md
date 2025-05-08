# Egocentric Point Tracking

This repository contains the implementation of a point tracking system for egocentric videos using Foundational Models like DINOv2 and CroCov2.

## EgoPoints Dataset

The EgoPoints dataset can be downloaded online [here](https://ahmaddarkhalil.github.io/EgoPoints/index.html#sec7). This provides sparse point annotations for egocentric egocentric sequences.


## Model Architecture

Our model architecture is defined in `models/model.py`. Key components include:

- Feature extraction using DINOv2 (loaded from torch hub)
- CroCov2 model encoder for feature extraction
- Custom delta networks for improved point tracking


## Training

To train the model:

```bash
# Example training with DINOv2 backbone and delta network
sbatch train_dino3d_ego_points3d_combined_loss_delta.sh
```

Training parameters include:
- Batch size: 32
- Learning rate: 1e-6
- 100 epochs with cosine decay schedule
- Combined loss function (spatial + feature similarity)

The `trainer.py` file contains the core training logic, including:
- Data loading and batch processing
- Model optimization
- Validation loops
- Checkpointing and logging

## Evaluation

Evaluate point tracking performance using:

```bash
python point_tracking_evaluation.py --model_path /path/to/model --eval_videos evaluation_paths.json
```

This script processes a set of videos defined in a JSON file and outputs tracking accuracy metrics for the model.

## Visualization

To visualize point tracking results:

```bash
python visualize.py --model_path /path/to/model --video_path /path/to/video
```


## Acknowledgments
We thank Ahmad Dharkalil for his help with egopoints data
