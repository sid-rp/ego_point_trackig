import os
import math
import argparse
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

from datasets import EpicKitchenDataset
from models import DinoDeltaModel, CrocoDeltaNet, CrocoF, DinoF
from utils.losses import *
from utils import misc


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def bilinear_sample(feat_map, keypoints):
    """
    Bilinearly sample features at subpixel keypoint locations.
    
    Arguments:
    feat_map -- Feature map of shape (1, C, H, W)
    keypoints -- Keypoints tensor of shape (N, 2) in pixel coordinates (x, y)
    
    Returns:
    sampled_feats -- Feature vectors of shape (N, C)
    """
    # breakpoint()
    # breakpoint()
    N, _ = keypoints.shape

    H, W = feat_map.shape[2], feat_map.shape[3]

    # Normalize keypoints to [-1, 1] for grid_sample
    norm_x = (keypoints[:, 0] / (W - 1)) * 2 - 1
    norm_y = (keypoints[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack((norm_x, norm_y), dim=1).view(1, N, 1, 2)  # (1, N, 1, 2)

    # Sample the features using bilinear interpolation
    sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # (1, C, N, 1)
    # breakpoint()
    return sampled.squeeze(-1).squeeze(0).permute(1, 0)  # (N, C)


def find_nearest_neighbors(feat_px1, feat_px2, keypoints1, k=1):
    """
    Find k nearest neighbors using cosine similarity 
    """
    B, C, H, W = feat_px2.shape

    # Sample features at keypoints1
    feat_px1_kps = bilinear_sample(feat_px1, keypoints1)  # (N, C)

    # Flatten and normalize feat_px2
    feat_px2_flat = feat_px2.view(C, -1).permute(1, 0)  # (H*W, C)

    # Normalize features
    feat_px1_kps = F.normalize(feat_px1_kps, dim=1)
    feat_px2_flat = F.normalize(feat_px2_flat, dim=1)

    # Cosine similarity and distances
    sim = torch.matmul(feat_px1_kps, feat_px2_flat.T)  # (N, H*W)
    # breakpoint()
    dists = 1.0 - sim  # cosine distance

    # Top-k nearest indices (highest similarity)
    topk_indices = torch.topk(sim, k, dim=1, largest=True).indices  # (N, k)

    # Convert to (x, y) coordinates
    y = torch.div(topk_indices, W, rounding_mode='floor')
    x = topk_indices % W
    neighbors_coordinates = [torch.stack([x[i], y[i]], dim=1) for i in range(topk_indices.shape[0])]

    return topk_indices, neighbors_coordinates


def get_valid_frames(valids):
    valid_frames = []
    for point_idx in range(valids.shape[1]):
        valid_frames.extend(np.where(valids[:, point_idx] == 1.0)[0].tolist())
    return sorted(set(valid_frames))



def load_valid_frames(root, valid_frame):
    img_path = f"{root}/frame_{str(valid_frame+1).zfill(10)}.jpg"
    img = Image.open(img_path)
    return np.array(img)


def processed_image(image):

    img_tensor  = test_transform(image)
    # breakpoint()

    return img_tensor

def inference_loop(processed_pair, model, device="cpu"):
    with torch.no_grad():
        tgt_tensor, src_tensor = processed_pair
        tgt_tensor = tgt_tensor.unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)
        src_tensor = src_tensor.unsqueeze(0).to(device)
        feat_target, feat_source  = model(tgt_tensor, src_tensor)
    return feat_target, feat_source

def rescale_keypoints(trajs_gt, original_shape, target_shape=(224, 224)):
    """
    Rescales 2D keypoints from original image size to target image size.
    
    Parameters:
        trajs_gt (np.ndarray): Array of shape (..., 2), where last dim is (x, y)
        original_shape (tuple): (height, width) of original image
        target_shape (tuple): (height, width) of resized image, default (224, 224)

    Returns:
        np.ndarray: Rescaled keypoints
    """
    orig_h, orig_w = original_shape
    target_h, target_w = target_shape
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    trajs_gt[..., 0] *= scale_x  # x-coordinates
    trajs_gt[..., 1] *= scale_y  # y-coordinates
    return trajs_gt

def compute_tracking_metrics(pred_kps, gt_kps, thresholds=[2, 4, 6, 8,16, 24]):
    """
    pred_kps: (N, 2) array of predicted keypoints
    gt_kps: (N, 2) array of ground-truth keypoints
    thresholds: list of pixel thresholds
    """
    errors = np.linalg.norm(pred_kps - gt_kps, axis=1)  # (N,)
    metrics = {}
    metrics['EPE'] = np.mean(errors)
    
    for th in thresholds:
        acc = np.mean(errors < th)
        # breakpoint()
        metrics[f'Accuracy@{th}px'] = acc

    return metrics, errors

def evaluate_video(root, model, frames, gt_tracks, initial_keypoints, thresholds=[2, 4, 8, 12, 16, 20, 24]):
    """
    model: your tracking model
    frames: list of video frames [f0, f1, ..., fT]
    gt_tracks: list of ground-truth keypoints per frame [(N, 2) arrays]
    initial_keypoints: (N, 2) array of initial keypoints at frame 0
    thresholds: thresholds to compute Accuracy
    
    Returns:
        video_metrics: dict of average metrics over the video
    """
    num_frames = len(frames)
    pred_keypoints = initial_keypoints.copy()  # Current keypoints to track

    all_errors = []

    for t in tqdm(range(1, num_frames), desc="Tracking frames"):
        # Predict keypoints in frame t given frame t-1
        frame_1 = load_valid_frames(root, frames[t-1])
        frame_2 = load_valid_frames(root, frames[t])
        frame_1 = processed_image(frame_1)
        frame_2 = processed_image(frame_2)
      


        feat1, feat2 = inference_loop((frame_1, frame_2), model)

        # Use the predicted keypoints from last frame (not initial ones!)
        pred_kpts_tensor = torch.from_numpy(pred_keypoints).to(torch.float32)  # (N, 2)

        _, new_pred_keypoints = find_nearest_neighbors(feat1, feat2, pred_kpts_tensor, k=1)

        # Update predicted keypoints
        # breakpoint()
        new_pred_keypoints = torch.stack(new_pred_keypoints, dim = 0)
        # breakpoint()
        
        pred_keypoints = new_pred_keypoints.squeeze().cpu().numpy()  # (N, 2)

        # Ground-truth keypoints at frame t
        gt_keypoints = gt_tracks[t]

        # Compute per-frame tracking metrics
        _, errors = compute_tracking_metrics(pred_keypoints, gt_keypoints, thresholds)
        all_errors.append(errors)
    
    # Aggregate metrics
    all_errors = np.concatenate(all_errors)  # (N * (T-1),)
    video_metrics = {}
    video_metrics['EPE'] = np.mean(all_errors)
    for th in thresholds:
        video_metrics[f'Accuracy@{th}px'] = np.mean(all_errors < th)

    # Print metrics nicely
    print("\n--- Video Tracking Metrics ---")
    print(f"Mean EPE: {video_metrics['EPE']:.4f}")
    for th in thresholds:
        print(f"Accuracy @{th}px: {video_metrics[f'Accuracy@{th}px']*100:.2f}%")
    print("--------------------------------\n")
    
    return video_metrics




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_location", type=str, help="Path to EgoPoints folder", default="/scratch/projects/fouheylab/shared_datasets/point_tracking_data/ego_points")
    parser.add_argument("--seq_name", type=str, help="EgoPoints sequence name to visualise", default="P08_21_start_21480_end_22217")
    args = parser.parse_args()
    # print(f"Visualising tracks for {args.seq_name}...")

    # Load annotations
    annots = np.load(f"{args.dataset_location}/{args.seq_name}/annot.npz")

    valids = annots["valids"]
    valid_frames = get_valid_frames(valids)
    trajs_gt = annots["trajs_2d"][valid_frames, :, :]

    device = "cpu"
    MODEL_WEIGHTS = "/scratch/projects/fouheylab/dma9300/OSNOM/croco_model_epochs_70_combined_loss_deltanet/model_loss_combined_epoch_70_val_loss_0.0338.pth"

    model = CrocoDeltaNet(delta=True)
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    root  =  f"{args.dataset_location}/{args.seq_name}/rgbs"

    frame_1 = load_valid_frames(root, valid_frames[0])
    # breakpoint()
    trajs_gt = rescale_keypoints(trajs_gt, frame_1.shape[:2])
    # breakpoint()

    evaluate_video(root, model,valid_frames, trajs_gt,trajs_gt[0])
    breakpoint()












# def load_image_pairs(root, valid_frames, target_size=(224, 224)):
#     # Load target image (the first in the list)
#     target_idx = valid_frames[0]
#     target_path = f"{root}/frame_{str(target_idx + 1).zfill(10)}.jpg"
#     target_img = Image.open(target_path).resize(target_size, Image.BILINEAR)

#     # Generate image pairs (target, source)
#     image_pairs = []
#     for source_idx in valid_frames[0:]:
#         source_path = f"{root}/frame_{str(source_idx + 1).zfill(10)}.jpg"
#         source_img = Image.open(source_path)
#         image_pairs.append((target_img.copy(), source_img))  # Use .copy() to avoid reference issues

#     return image_pairs

# def images_loader(root, valid_frames):
#     image_pairs = []
#     for source_idx in valid_frames:
#         source_path = f"{root}/frame_{str(source_idx + 1).zfill(10)}.jpg"
#         source_img = Image.open(source_path)
#         image_pairs.append(source_img) 
#     return image_pairs



# def processed_pairs(image_pairs):
#     processed_pairs = []
#     for tgt_img, src_img in image_pairs:
#         tgt_tensor = test_transform(np.array(tgt_img))
#         src_tensor = test_transform(np.array(src_img))
#         processed_pairs.append((tgt_tensor, src_tensor))
#     return processed_pairs

# def inference_loop(processed_pair, model, device="cpu"):
#     with torch.no_grad():
#         tgt_tensor, src_tensor = processed_pair
#         tgt_tensor = tgt_tensor.unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)
#         src_tensor = src_tensor.unsqueeze(0).to(device)
#         feat_target, feat_source  = model(tgt_tensor, src_tensor)
#     return feat_target, feat_source
        


# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset_location", type=str, help="Path to EgoPoints folder", default="/scratch/projects/fouheylab/shared_datasets/point_tracking_data/ego_points")
# parser.add_argument("--seq_name", type=str, help="EgoPoints sequence name to visualise", default="P08_21_start_21480_end_22217")
# args = parser.parse_args()
# print(f"Visualising tracks for {args.seq_name}...")

# # Load annotations
# annots = np.load(f"{args.dataset_location}/{args.seq_name}/annot.npz")
# trajs_gt = annots["trajs_2d"]
# valids = annots["valids"]
# valid_frames = get_valid_frames(valids)



# # Load concatenated frames (224x224 each)
# concat_frames, img_shape = load_valid_frames(f"{args.dataset_location}/{args.seq_name}/rgbs", valid_frames, target_size=(224, 224))
# trajs_gt = rescale_keypoints(trajs_gt,img_shape)


# # Load and preprocess image pairs
# image_pairs = load_image_pairs(f"{args.dataset_location}/{args.seq_name}/rgbs", valid_frames)
# # breakpoint()
# processed_pairs = processed_pairs(image_pairs)
# images  = images_loader(f"{args.dataset_location}/{args.seq_name}/rgbs", valid_frames)
# # model_vis = DinoF()
# # visualize_dino_features(images, model_vis, save_path="results.jpg", img_size=224, mask_threshold=0.0)
# # breakpoint()

# Initialize model
# device = "cpu"
# MODEL_WEIGHTS = "/scratch/projects/fouheylab/dma9300/OSNOM/dino_model_epochs_100_ego_points_l2_loss_no_delta/model_loss_l2_epoch_40_val_loss_0.0176.pth"

# model = DinoDeltaModel(delta=True)
# checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
# model.load_state_dict(checkpoint)
# model.to(device)
# model.eval()

# # Set up plot
# img_shape = (224, 224)
# fig, axs = plt.subplots(1, 1, figsize=(8, 3))
# axs.axis("off")
# axs.imshow(concat_frames)
# offset_x = img_shape[1]

# plot_lines = {}

# for frame_num, frame_idx in enumerate(valid_frames):
#     processed_pair = processed_pairs[frame_num]

#     for point_idx in range(valids.shape[1]):
#         if valids[frame_idx, point_idx] != 1.0:
#             continue

#         if point_idx not in plot_lines:
#             plot_lines[point_idx] = {"coords": []}

#         # Get target keypoint from GT and wrap in batch
#         kpts1 = trajs_gt[frame_idx, point_idx]  # (2,)
#         kpts1_tensor = torch.from_numpy(kpts1).to(torch.float32).unsqueeze(0)  # (1, 1, 2)

#         # Run model
#         feat1, feat2 = inference_loop(processed_pair, model)

#         # Track to next frame
#         if frame_num == 0:
#             x = kpts1[0]
#             y = kpts1[1]
#         else:
#             _, kpts2 = find_nearest_neighbors(feat1, feat2, kpts1_tensor, k=1)
#             x = kpts2[0][0, 0]
#             y = kpts2[0][0, 1]

#         # Store tracked coordinates with image offset
#         plot_lines[point_idx]["coords"].append([(x + (offset_x * frame_num)), y])

#         # Optional: linestyle for dynamic/static objects
#         if "dynamic_obj_tracks" in annots:
#             is_dynamic = annots["dynamic_obj_tracks"][point_idx] == 1
#             plot_lines[point_idx]["linestyle"] = ":" if is_dynamic else "-"
#         else:
#             plot_lines[point_idx]["linestyle"] = "-"

# # Draw all tracked trajectories
# for point_idx in plot_lines:
#     coords = np.array(plot_lines[point_idx]["coords"])
#     axs.plot(coords[:, 0], coords[:, 1], linestyle=plot_lines[point_idx]["linestyle"], marker="o")

# fig.savefig(f"track_vis_{args.seq_name}_DINO_newwww.png", bbox_inches="tight")
# plt.close(fig)
