import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
from datasets import EpicKitchenDataset
from models import DinoDeltaModel, CrocoDeltaNet
from utils.losses import * 
from utils import misc
from collections import defaultdict
import argparse
from torch.cuda.amp import autocast, GradScaler
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity



def tensor_to_uint8(img_tensor):
    """
    Unnormalize a normalized image tensor and convert it to uint8 NumPy array.
    Args:
        img_tensor: (3, H, W) tensor normalized with ImageNet stats
    Returns:
        img_uint8: (H, W, 3) uint8 NumPy array in [0, 255]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(-1, 1, 1)
    img = img_tensor * std + mean  # Unnormalize
    img = img.clamp(0, 1)          # Clamp to valid range
    img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img_uint8 = (img * 255).round().astype(np.uint8)
    return img_uint8


def normalize_img(img):
    """Normalize an image to [0, 1] range."""
    img = img - img.min()
    img = img / (img.max() + 1e-5)
    return img

def reduce_features_to_rgb(feats):
    """
    Apply PCA to reduce the feature map to 3 channels for visualization.
    """
    # Reshape feature maps from (1, 768, 224, 224) to (768, 224*224) for PCA
    feats = feats.squeeze(0)  # Remove batch dimension (1, 768, 224, 224) -> (768, 224, 224)
    feats_reshaped = feats.view(768, -1).transpose(0, 1)  # Shape: (224*224, 768)
    
    # Apply PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    reduced_feats = pca.fit_transform(feats_reshaped)
    
    # Reshape back to image dimensions (224, 224, 3)
    reduced_feats = reduced_feats.reshape(224, 224, 3)
    return torch.tensor(reduced_feats).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)

def bilinear_sample(feat_map, keypoints):
    """
    Bilinearly sample features at subpixel keypoint locations.
    
    Arguments:
    feat_map -- Feature map of shape (1, C, H, W)
    keypoints -- Keypoints tensor of shape (N, 2) in pixel coordinates (x, y)
    
    Returns:
    sampled_feats -- Feature vectors of shape (N, C)
    """
    N, _ = keypoints.shape
    H, W = feat_map.shape[2], feat_map.shape[3]

    # Normalize keypoints to [-1, 1] for grid_sample
    norm_x = (keypoints[:, 0] / (W - 1)) * 2 - 1
    norm_y = (keypoints[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack((norm_x, norm_y), dim=1).view(1, N, 1, 2)  # (1, N, 1, 2)

    # Sample the features using bilinear interpolation
    sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # (1, C, N, 1)
    return sampled.squeeze(-1).squeeze(0).permute(1, 0)  # (N, C)

import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def find_nearest_neighbors(feat_px1, feat_px2, keypoints1, k=1, distance_threshold=0.30):
    """
    Find the k nearest neighbors in feat_px2 for keypoints in feat_px1 using cosine similarity.
    
    Returns:
    - nearest_neighbors: Indices of the k nearest neighbors in feat_px2 for each keypoint in feat_px1.
    - neighbors_coordinates: Corresponding (x, y) coordinates for top-k neighbors.
    - visibility_mask: Boolean mask (224, 224), True at keypoints where distance < threshold.
    """
    device = feat_px1.device
    feat_px1_kps = bilinear_sample(feat_px1, keypoints1)  # (N, 768)

    # Flatten feat_px2 for similarity comparison
    B, C, H, W = feat_px2.shape
    feat_px2_flat = feat_px2.view(C, -1).permute(1, 0).cpu().numpy()  # (H*W, 768)

    # Cosine similarity and distances
    sim = cosine_similarity(feat_px1_kps.cpu().numpy(), feat_px2_flat)  # (N, H*W)
    sim_tensor = torch.tensor(sim, device=device)  # back to torch
    distances = 1 - sim_tensor  # (N, H*W)

    # Top-k indices (for k=1 only below)
    topk_indices = torch.argsort(sim_tensor, dim=1, descending=True)[:, :k]  # (N, k)

    # Top-1 match distances
    top1_indices = topk_indices[:, 0]  # (N,)
    top1_distances = distances.gather(1, top1_indices.unsqueeze(1)).squeeze(1)  # (N,)

    # Visibility mask
    visibility_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    valid_mask = top1_distances < distance_threshold
    valid_keypoints = keypoints1[valid_mask]  # (M, 2)

    # Round to nearest int pixel and clamp to bounds
    valid_keypoints_rounded = torch.round(valid_keypoints).long()
    valid_keypoints_rounded[:, 0].clamp_(0, W - 1)
    valid_keypoints_rounded[:, 1].clamp_(0, H - 1)

    # Set mask at those locations to True
    visibility_mask[valid_keypoints_rounded[:, 1], valid_keypoints_rounded[:, 0]] = True

    # Convert top-k indices to coordinates
    neighbors_coordinates = []
    for i in range(topk_indices.shape[0]):
        indices = topk_indices[i]
        y = torch.div(indices, W, rounding_mode='floor')
        x = indices % W
        coords = torch.stack([x, y], dim=1)
        neighbors_coordinates.append(coords)

    return topk_indices, neighbors_coordinates, visibility_mask


def visualize_model_outputs(image1, image2, feat_px1, feat_px2, keypoints1, keypoints2, kp1_mask, kp2_mask, k=1):
    """
    Visualize the reduced features, keypoints, nearest neighbors, and visibility masks.
    """
    # Reduce features for visualization
    reduced_feat_px1 = reduce_features_to_rgb(feat_px1)
    reduced_feat_px2 = reduce_features_to_rgb(feat_px2)

    keypoints1 = keypoints1.squeeze(0)
    keypoints2 = keypoints2.squeeze(0)

    keypoints1 = keypoints1[kp1_mask.squeeze()]
    keypoints2 = keypoints2[kp2_mask.squeeze()]

    feat_px1 = F.normalize(feat_px1, dim=-1) 
    feat_px2 = F.normalize(feat_px2, dim=-1) 
  
    nearest_neighbors2, coordinates2, visibility_mask2 = find_nearest_neighbors(feat_px1, feat_px2, keypoints1, k)
    nearest_neighbors1, coordinates1, visibility_mask1 = find_nearest_neighbors(feat_px2, feat_px1, keypoints2, k)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    colors = plt.cm.get_cmap('hsv', len(keypoints1))

    # Image 1 with Keypoints
    axes[0, 0].imshow(tensor_to_uint8(image1.squeeze(0)))
    for i, kp in enumerate(keypoints1):
        axes[0, 0].scatter(kp[0], kp[1], color=colors(i), marker='x')
    axes[0, 0].set_title('Image 1 with Keypoints 1')
    axes[0, 0].axis('off')

    # Image 2 with Keypoints
    axes[0, 1].imshow(tensor_to_uint8(image2.squeeze(0)))
    for i, kp in enumerate(keypoints2):
        axes[0, 1].scatter(kp[0], kp[1], color=colors(i), marker='x')
    axes[0, 1].set_title('Image 2 with Keypoints 2')
    axes[0, 1].axis('off')

    # Reduced Feature Map 1 with matches
    axes[1, 0].imshow(reduced_feat_px1.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[1, 0].set_title('Reduced Feature Map (feat_px1)')
    for i, (kp, nn_coords) in enumerate(zip(keypoints1, coordinates1)):
        axes[1, 0].scatter(kp[0], kp[1], color=colors(i), marker='x')
        for nn_x, nn_y in nn_coords:
            axes[1, 0].scatter(nn_x, nn_y, color=colors(i), marker='o')
            axes[1, 0].plot([kp[0], nn_x], [kp[1], nn_y], color=colors(i), linestyle='--', linewidth=1)
    axes[1, 0].axis('off')

    # Reduced Feature Map 2 with matches
    axes[1, 1].imshow(reduced_feat_px2.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[1, 1].set_title('Reduced Feature Map (feat_px2)')
    for i, (kp, nn_coords) in enumerate(zip(keypoints2, coordinates2)):
        axes[1, 1].scatter(kp[0], kp[1], color=colors(i), marker='x')
        for nn_x, nn_y in nn_coords:
            axes[1, 1].scatter(nn_x, nn_y, color=colors(i), marker='o')
            axes[1, 1].plot([kp[0], nn_x], [kp[1], nn_y], color=colors(i), linestyle='--', linewidth=1)
    axes[1, 1].axis('off')

    # Visibility mask for keypoints1 → feat_px2
    axes[0, 2].imshow(visibility_mask2.cpu().numpy(), cmap='gray')
    axes[0, 2].set_title('Visibility Mask (kp1 → feat_px2)')
    axes[0, 2].axis('off')

    # Visibility mask for keypoints2 → feat_px1
    axes[1, 2].imshow(visibility_mask1.cpu().numpy(), cmap='gray')
    axes[1, 2].set_title('Visibility Mask (kp2 → feat_px1)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig("dino_visualization_vis.png")
    breakpoint()
    plt.show()



def visualizer(args):
    def worker_init_fn(worker_id):
        dataset = torch.utils.data.get_worker_info().dataset
        dataset._load_and_cache_all_tar_files()
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    
    # Create datasets
    train_dataset = EpicKitchenDataset(root=args.root, overlap_threshold=args.overlap_threshold, split="val")
    # val_dataset = EpicKitchenDataset(root=args.root, overlap_threshold=args.overlap_threshold, split="val")

    
    # Create DataLoaders
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn = misc.custom_collate_fn, worker_init_fn=worker_init_fn)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn = misc.custom_collate_fn, worker_init_fn=worker_init_fn)


    # Initialize model and optimizer
    model = CrocoDeltaNet(delta =args.use_delta)


    # Load weights
    checkpoint_path = args.checkpoint_path  # ← replace this path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # If you saved the full model state_dict:
    model.load_state_dict(checkpoint)
    model.eval()

    for batch in dataloader:
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        kp1 = batch['kp1'].to(device)
        kp2 = batch['kp2'].to(device)
        kp1_mask  = batch["kp1_mask"].to(device)
        kp2_mask  = batch["kp2_mask"].to(device)

        kp1_mask = batch.get('kp1_mask', None)
        if kp1_mask is not None:
            kp1_mask = kp1_mask.to(device)

        with torch.cuda.amp.autocast(enabled=True):
            # Forward pass
            with torch.no_grad():
                feat1, feat2 = model(img1, img2)
                # feat2_combined = feat2 + delta

                # visualize_dino_feats_and_keypoints(img1.squeeze(), img2.squeeze(), feat1, feat2_combined, kp1.squeeze(),kp2.squeeze(), kp1_mask.squeeze(), kp2_mask.squeeze())
                visualize_model_outputs(img1, img2, feat1, feat2, kp1, kp2, kp1_mask, kp2_mask)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize keypoint matching for DinoDeltaModel on EpicKitchenDataset")

    # Dataset arguments
    parser.add_argument('--root', type=str, default='/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/',
                        help="Root directory for datasets")
    parser.add_argument('--overlap_threshold', type=float, default=0.2,
                        help="Threshold for camera frustum overlap")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for training and validation")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of workers for DataLoader")

    # Training arguments (only needed for completeness in this script)
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--print_freq', type=int, default=10,
                        help="Frequency of printing training stats")
    parser.add_argument('--write_freq', type=int, default=100,
                        help="Frequency of writing stats to tensorboard")
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help="Number of warm-up epochs to increase learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help="Minimum learning rate")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the model checkpoints")
    parser.add_argument('--use_delta', type=bool, default=False, help = "use delta net")

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use (cuda or cpu)")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the pretrained model checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    visualizer(args)