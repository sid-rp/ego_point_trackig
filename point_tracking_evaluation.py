# import os
# import math
# import argparse
# import random
# import json

# from collections import defaultdict

# import numpy as np
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp import autocast, GradScaler
# import torchvision.transforms as T
# from torchvision import transforms
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from PIL import Image
# from tqdm import tqdm
# from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import minmax_scale

# from datasets import EpicKitchenDataset
# from models import DinoDeltaModel, CrocoDeltaNet, CrocoF, DinoF
# from utils.losses import *
# from utils import misc


# test_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# def bilinear_sample(feat_map, keypoints):
#     """
#     Bilinearly sample features at subpixel keypoint locations.
    
#     Arguments:
#     feat_map -- Feature map of shape (1, C, H, W)
#     keypoints -- Keypoints tensor of shape (N, 2) in pixel coordinates (x, y)
    
#     Returns:
#     sampled_feats -- Feature vectors of shape (N, C)
#     """
#     # breakpoint()
#     # breakpoint()
#     N, _ = keypoints.shape

#     H, W = feat_map.shape[2], feat_map.shape[3]

#     # Normalize keypoints to [-1, 1] for grid_sample
#     norm_x = (keypoints[:, 0] / (W - 1)) * 2 - 1
#     norm_y = (keypoints[:, 1] / (H - 1)) * 2 - 1
#     grid = torch.stack((norm_x, norm_y), dim=1).view(1, N, 1, 2)  # (1, N, 1, 2)

#     # Sample the features using bilinear interpolation
#     sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # (1, C, N, 1)
#     # breakpoint()
#     return sampled.squeeze(-1).squeeze(0).permute(1, 0)  # (N, C)


# # def find_nearest_neighbors(feat_px1, feat_px2, keypoints1, k=1):
# #     """
# #     Find k nearest neighbors using cosine similarity 
# #     """
# #     B, C, H, W = feat_px2.shape

# #     # Sample features at keypoints1
# #     feat_px1_kps = bilinear_sample(feat_px1, keypoints1)  # (N, C)

# #     # Flatten and normalize feat_px2
# #     feat_px2_flat = feat_px2.view(C, -1).permute(1, 0)  # (H*W, C)

# #     # Normalize features
# #     feat_px1_kps = F.normalize(feat_px1_kps, dim=1)
# #     feat_px2_flat = F.normalize(feat_px2_flat, dim=1)

# #     # Cosine similarity and distances
# #     sim = torch.matmul(feat_px1_kps, feat_px2_flat.T)  # (N, H*W)
# #     # breakpoint()
# #     dists = 1.0 - sim  # cosine distance

# #     # Top-k nearest indices (highest similarity)
# #     topk_indices = torch.topk(sim, k, dim=1, largest=True).indices  # (N, k)

# #     # Convert to (x, y) coordinates
# #     y = torch.div(topk_indices, W, rounding_mode='floor')
# #     x = topk_indices % W
# #     neighbors_coordinates = [torch.stack([x[i], y[i]], dim=1) for i in range(topk_indices.shape[0])]

# #     return topk_indices, neighbors_coordinates

# import torch.nn.functional as F

# def find_nearest_neighbors(feat_px1, feat_px2, keypoints1, k=1, window_size=2):
#     """
#     Find k nearest neighbors with subpixel precision using cosine similarity and soft-argmax.
#     """
#     B, C, H, W = feat_px2.shape

#     # Sample features at keypoints1
#     feat_px1_kps = bilinear_sample(feat_px1, keypoints1)  # (N, C)

#     # Flatten and normalize feat_px2
#     feat_px2_flat = feat_px2.view(C, -1).permute(1, 0)  # (H*W, C)

#     # Normalize features
#     feat_px1_kps = F.normalize(feat_px1_kps, dim=1)
#     feat_px2_flat = F.normalize(feat_px2_flat, dim=1)

#     # Cosine similarity
#     sim = torch.matmul(feat_px1_kps, feat_px2_flat.T)  # (N, H*W)

#     # Get top-1 match index (integer pixel)
#     top1_indices = torch.topk(sim, 1, dim=1, largest=True).indices.squeeze(1)  # (N,)

#     # Coarse integer (x, y)
#     coarse_y = torch.div(top1_indices, W, rounding_mode='floor')
#     coarse_x = top1_indices % W

#     # Soft-argmax refinement
#     refined_keypoints = []

#     pad = window_size // 2
#     sim_maps = sim.view(-1, H, W)  # (N, H, W)

#     for i in range(sim_maps.shape[0]):
#         cx = coarse_x[i]
#         cy = coarse_y[i]

#         # Crop local window
#         x0 = torch.clamp(cx - pad, 0, W - 1)
#         x1 = torch.clamp(cx + pad + 1, 0, W)
#         y0 = torch.clamp(cy - pad, 0, H - 1)
#         y1 = torch.clamp(cy + pad + 1, 0, H)

#         window = sim_maps[i, y0:y1, x0:x1]  # (small h, small w)

#         # Create coordinate grid
#         yy, xx = torch.meshgrid(
#             torch.arange(y0, y1, device=sim.device),
#             torch.arange(x0, x1, device=sim.device),
#             indexing="ij"
#         )

#         window = window.flatten()
#         xx = xx.flatten()
#         yy = yy.flatten()

#         # Softmax over the window
#         prob = F.softmax(window, dim=0)

#         # Compute expected (x, y) coordinate
#         refined_x = (prob * xx).sum()
#         refined_y = (prob * yy).sum()

#         refined_keypoints.append(torch.stack([refined_x, refined_y], dim=0))

#     refined_keypoints = torch.stack(refined_keypoints, dim=0)  # (N, 2)

#     return top1_indices, refined_keypoints


# def get_valid_frames(valids):
#     valid_frames = []
#     for point_idx in range(valids.shape[1]):
#         valid_frames.extend(np.where(valids[:, point_idx] == 1.0)[0].tolist())
#     return sorted(set(valid_frames))



# def load_frames(root, valid_frame):
#     img_path = f"{root}/{valid_frame}"
#     img = Image.open(img_path)
#     return np.array(img)


# def processed_image(image):

#     img_tensor  = test_transform(image)
#     # breakpoint()

#     return img_tensor

# def inference_loop(processed_pair, model, device="cpu"):
#     with torch.no_grad():
#         tgt_tensor, src_tensor = processed_pair
#         tgt_tensor = tgt_tensor.unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)
#         src_tensor = src_tensor.unsqueeze(0).to(device)
#         feat_target, feat_source  = model(tgt_tensor, src_tensor)
#     return feat_target, feat_source

# def rescale_keypoints(trajs_gt, original_shape, target_shape=(224, 224)):
#     """
#     Rescales 2D keypoints from original image size to target image size.
    
#     Parameters:
#         trajs_gt (np.ndarray): Array of shape (..., 2), where last dim is (x, y)
#         original_shape (tuple): (height, width) of original image
#         target_shape (tuple): (height, width) of resized image, default (224, 224)

#     Returns:
#         np.ndarray: Rescaled keypoints
#     """
#     orig_h, orig_w = original_shape
#     target_h, target_w = target_shape
#     scale_x = target_w / orig_w
#     scale_y = target_h / orig_h
#     trajs_gt[..., 0] *= scale_x  # x-coordinates
#     trajs_gt[..., 1] *= scale_y  # y-coordinates
#     return trajs_gt


# def compute_tracking_metrics(pred_kps, gt_kps, mask, thresholds=[1, 2, 4, 8, 16, 24]):
#     """
#     pred_kps: (N, 2) array of predicted keypoints
#     gt_kps: (N, 2) array of ground-truth keypoints
#     mask: (N,) boolean array indicating valid keypoints
#     thresholds: list of pixel thresholds
#     """
#     # Apply the mask
#     pred_kps = pred_kps[mask]
#     gt_kps = gt_kps[mask]

#     if len(pred_kps) == 0:
#         # No valid keypoints, return NaNs
#         metrics = {f'Accuracy@{th}px': np.nan for th in thresholds}
#         metrics['EPE'] = np.nan
#         metrics['delta_avg'] = np.nan
#         errors = np.array([])
#         return metrics, errors

#     errors = np.linalg.norm(pred_kps - gt_kps, axis=1)  # (N,)
#     metrics = {}
#     metrics['EPE'] = np.mean(errors)
    
#     accuracies = []
#     for th in thresholds:
#         acc = np.mean(errors <= th)
#         metrics[f'Accuracy@{th}px'] = acc
#         accuracies.append(acc)

#     # Compute δavg over {1,2,4,8,16}
#     selected_accuracies = [metrics[f'Accuracy@{th}px'] for th in [1, 2, 4, 8, 16] if f'Accuracy@{th}px' in metrics]
#     if selected_accuracies:
#         metrics['delta_avg'] = np.mean(selected_accuracies)
#     else:
#         metrics['delta_avg'] = np.nan  # fallback

#     return metrics, errors


# def evaluate_video(root,annots, model, frames,thresholds=[8, 16, 24]):
#     trajs_gt = annots["trajs_2d"]
#     frame_1 = load_frames(root, frames[0])
#     trajs_gt = rescale_keypoints(trajs_gt, frame_1.shape[:2])

#     valids  = annots["valids"]
#     out_of_view = annots["out_of_view"]
#     occluded  = annots["occluded"]
#     visibs  = annots["visibs"]
#     visibs_valid  = annots["vis_valids"]


#     initial_keypoints = trajs_gt[0, :,:]
#     gt_tracks = trajs_gt

#     num_frames = len(frames)
#     pred_keypoints = initial_keypoints.copy()

#     all_errors = []

#     for t in tqdm(range(1, num_frames), desc="Tracking frames"):
#         frame_1 = load_frames(root, frames[t-1])
#         frame_2 = load_frames(root, frames[t])

#         frame_1 = processed_image(frame_1)
#         frame_2 = processed_image(frame_2)

#         feat1, feat2 = inference_loop((frame_1, frame_2), model)

#         pred_kpts_tensor = torch.from_numpy(pred_keypoints).to(torch.float32)
#         _, new_pred_keypoints = find_nearest_neighbors(feat1, feat2, pred_kpts_tensor, k=1)

#         # breakpoint()

#         new_pred_keypoints = new_pred_keypoints
#         pred_keypoints = new_pred_keypoints.squeeze().cpu().numpy()

#         gt_keypoints = gt_tracks[t, :, : ]
#         mask = ( valids[t, :].astype(bool) & (~out_of_view[t, :].astype(bool)) & (~occluded[t, :].astype(bool)) & visibs[t, :].astype(bool) & visibs_valid[t, :].astype(bool))
        
#         _, errors = compute_tracking_metrics(pred_keypoints, gt_keypoints, mask, thresholds)
#         all_errors.append(errors)
    
#     # breakpoint()
#     # Aggregate metrics
#     all_errors = np.concatenate(all_errors)
#     video_metrics = {}
#     video_metrics['EPE'] = np.mean(all_errors)
#     for th in thresholds:
#         video_metrics[f'Accuracy@{th}px'] = np.mean(all_errors < th)

#     # Compute δavg
#     selected_accuracies = [video_metrics[f'Accuracy@{th}px'] for th in thresholds if f'Accuracy@{th}px' in video_metrics]
#     if selected_accuracies:
#         video_metrics['delta_avg'] = np.mean(selected_accuracies)
#     else:
#         video_metrics['delta_avg'] = np.nan
    
#     return video_metrics



# def select_all_folders(root_path="/scratch/projects/fouheylab/shared_datasets/point_tracking_data/ego_points",
#                        output_json="evaluation_videos_all.json"):
#     """
#     Select all subfolders from a root directory and save their names to a JSON file.

#     Args:
#         root_path (str): Path to the root directory containing subfolders.
#         output_json (str): Path to save the JSON file.
#     """
#     # List all subfolders
#     all_folders = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]

#     if len(all_folders) == 0:
#         raise ValueError(f"No folders found in {root_path}.")

#     # Save all folders to a JSON file
#     with open(output_json, 'w') as f:
#         json.dump(all_folders, f, indent=2)

#     print(f"Saved {len(all_folders)} folders to {output_json}")


# if __name__ == "__main__":
#     # select_all_folders()
#     # breakpoint()
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_location", type=str, help="Path to EgoPoints folder", default="/scratch/projects/fouheylab/shared_datasets/point_tracking_data/ego_points")
#     parser.add_argument("--seq_list", type=str, help="Path to JSON file containing list of sequences", default="/scratch/projects/fouheylab/dma9300/OSNOM/evaluation_videos.json")
#     args = parser.parse_args()

#     import json
#     with open(args.seq_list, "r") as f:
#         sequence_list = json.load(f)  # sequence_list should be a list of sequence names

#     device = "cpu"
#     MODEL_WEIGHTS = "/scratch/projects/fouheylab/dma9300/OSNOM/croco_model_epochs_50_cosine_loss/model_loss_cosine_epoch_70_val_loss_0.0236.pth"

#     model = CrocoDeltaNet(delta = True)
#     checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
#     model.load_state_dict(checkpoint, strict=False)
#     model.to(device)
#     model.eval()

#     all_metrics = {}

#     for seq_name in tqdm(sequence_list, desc="Evaluating Sequences"):
#         print(f"Evaluating sequence: {seq_name}")

#         annots_path = os.path.join(args.dataset_location, seq_name, "annot.npz")
#         if not os.path.exists(annots_path):
#             print(f"Annotation file missing for {seq_name}, skipping...")
#             continue

#         annots = np.load(annots_path)
#         # breakpoint()
#         valids = annots["valids"]
#         # valid_frames = get_valid_frames(valids)
#         # trajs_gt = annots["trajs_2d"][valid_frames, :, :]

#         root = os.path.join(args.dataset_location, seq_name, "rgbs")

#         frames_names = sorted([fname for fname in os.listdir(root) if fname.endswith('.jpg') and fname.startswith('frame_')])
#         if not os.path.isdir(root):
#             print(f"RGB folder missing for {seq_name}, skipping...")
#             continue

#         # if len(valid_frames) < 2:
#         #     print(f"Not enough valid frames for {seq_name}, skipping...")
#         #     continue

       

#         # Run evaluation
#         # breakpoint()
#         metrics = evaluate_video(root,annots, model, frames_names)

#         all_metrics[seq_name] = metrics

#     # Print overall summary
#     print("\n\n--- Overall Evaluation Results ---")
#     all_epe = []
#     all_accs = defaultdict(list)  # Store accuracy@Xpx for each sequence
#     all_delta_avg = []  # Store delta_avg for each sequence

#     for seq, metrics in all_metrics.items():
#         print(f"{seq}: EPE={metrics['EPE']:.4f}")
#         all_epe.append(metrics['EPE'])
#         for k, v in metrics.items():
#             if k.startswith("Accuracy@"):
#                 all_accs[k].append(v)
#         if 'delta_avg' in metrics:
#             all_delta_avg.append(metrics['delta_avg'])

#     # Compute and print averages
#     if all_epe:
#         print(f"\nAverage EPE over all videos: {np.mean(all_epe):.4f}")
#         for acc_name, acc_values in all_accs.items():
#             avg_acc = np.mean(acc_values)
#             print(f"Average {acc_name} over all videos: {avg_acc*100:.2f}%")
#         if all_delta_avg:
#             avg_delta = np.mean(all_delta_avg)
#             print(f"Average δavg (accuracy @ [8,16, 24]px) over all videos: {avg_delta*100:.2f}%")

import os
import argparse
import json
from collections import defaultdict

from models.model import WEIGHTS_PATH, CrocoSSL,CrocoF,DinoF
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from models import CrocoDeltaNet, KeyPointNet
from utils.losses import *
from utils import misc


# --------------------------------------------
# Configuration
# --------------------------------------------

TEST_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# --------------------------------------------
# Utility Functions
# --------------------------------------------

def bilinear_sample(feat_map, keypoints):
    """Sample features at subpixel keypoint locations using bilinear interpolation."""
    N = keypoints.shape[0]
    H, W = feat_map.shape[2:]
    norm_x = (keypoints[:, 0] / (W - 1)) * 2 - 1
    norm_y = (keypoints[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack((norm_x, norm_y), dim=1).view(1, N, 1, 2)
    sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=False)
    return sampled.squeeze(-1).squeeze(0).permute(1, 0)  # (N, C)

def find_nearest_neighbors(feat1, feat2, keypoints, k=1):
    """Find k-nearest neighbors in feat2 for keypoints in feat1 using cosine similarity."""
    B, C, H, W = feat2.shape
    feat_kps = bilinear_sample(feat1, keypoints)
    feat2_flat = feat2.view(C, -1).permute(1, 0)
    feat_kps = F.normalize(feat_kps, dim=1)
    feat2_flat = F.normalize(feat2_flat, dim=1)
    sim = torch.matmul(feat_kps, feat2_flat.T)
    topk_indices = torch.topk(sim, k, dim=1).indices
    y = torch.div(topk_indices, W, rounding_mode='floor')
    x = topk_indices % W
    # breakpoint()
    return topk_indices, [torch.stack([x[i], y[i]], dim=1) for i in range(topk_indices.shape[0])]

def get_valid_frames(valids):
    """Extract unique frame indices that contain valid keypoints."""
    return sorted(set(np.where(valids == 1.0)[0].tolist()))

def load_image(root, idx):
    path = os.path.join(root, f"frame_{str(idx+1).zfill(10)}.jpg")
    return np.array(Image.open(path))

def process_image(img):
    return TEST_TRANSFORM(img)

def inference(model, img1, img2):
    with torch.no_grad():
        img1 = img1.unsqueeze(0).to(DEVICE)
        img2 = img2.unsqueeze(0).to(DEVICE)
        return model(img1, img2)

def rescale_keypoints(keypoints, original_shape, target_shape=(224, 224)):
    h_orig, w_orig = original_shape
    h_new, w_new = target_shape
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig
    keypoints[..., 0] *= scale_x
    keypoints[..., 1] *= scale_y
    return keypoints


def compute_metrics(pred_kps, gt_kps, thresholds=[8,16, 24]):
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


def evaluate_video(seq_root, model, frames, gt_tracks, init_keypoints, thresholds=[8, 16, 24]):
    pred_kps = init_keypoints.copy()
    all_errors = []

    for t in tqdm(range(1, len(frames)), desc="Tracking frames"):
        img1 = process_image(load_image(seq_root, frames[t - 1]))
        img2 = process_image(load_image(seq_root, frames[t]))
        feat1, feat2 = inference(model, img1, img2)

        pred_kps_tensor = torch.from_numpy(pred_kps).float()
        _, pred_kps_next = find_nearest_neighbors(feat1, feat2, pred_kps_tensor, k=1)
        pred_kps = torch.stack(pred_kps_next, dim=0).squeeze().cpu().numpy()

        _, errors = compute_metrics(pred_kps, gt_tracks[t], thresholds)
        all_errors.append(errors)

    all_errors = np.concatenate(all_errors)
    video_metrics = {"EPE": np.mean(all_errors)}
    for th in thresholds:
        video_metrics[f"Accuracy@{th}px"] = np.mean(all_errors < th)
    video_metrics["delta_avg"] = np.mean([video_metrics[f"Accuracy@{th}px"] for th in thresholds])
    return video_metrics


# --------------------------------------------
# Main
# --------------------------------------------

def main(args):
    with open(args.seq_list, "r") as f:
        sequences = json.load(f)

    model = DinoF()
    # model = CrocoSSL(model_weights=args.model_weights)
    # model = CrocoF(model_weights=args.model_weights)
    # model.load_state_dict(torch.load(args.model_weights, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()

    all_metrics = {}
    for seq in tqdm(sequences, desc="Evaluating Sequences"):
        print(f"\nEvaluating: {seq}")
        seq_root = os.path.join(args.dataset_location, seq, "rgbs")
        annot_path = os.path.join(args.dataset_location, seq, "annot.npz")

        if not os.path.exists(annot_path):
            print(f"Missing annotations for {seq}")
            continue

        annots = np.load(annot_path)
        valids = annots["valids"]
        valid_frames = get_valid_frames(valids)
        if len(valid_frames) < 2:
            print(f"Too few valid frames in {seq}")
            continue

        trajs = annots["trajs_2d"][valid_frames, :, :]
        first_img = load_image(seq_root, valid_frames[0])
        trajs = rescale_keypoints(trajs, first_img.shape[:2])
        # breakpoint()

        metrics = evaluate_video(seq_root, model, valid_frames, trajs, trajs[0])
        all_metrics[seq] = metrics

    print("\n--- Evaluation Summary ---")
    all_epe = [m["EPE"] for m in all_metrics.values()]
    acc_metrics = defaultdict(list)
    delta_avgs = []

    for m in all_metrics.values():
        for k, v in m.items():
            if k.startswith("Accuracy@"):
                acc_metrics[k].append(v)
        delta_avgs.append(m.get("delta_avg", np.nan))

    print(f"Mean EPE: {np.mean(all_epe):.4f}")
    for k, v in acc_metrics.items():
        print(f"{k}: {np.mean(v)*100:.2f}%")
    print(f"δavg: {np.nanmean(delta_avgs)*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_location", type=str, default="/scratch/projects/fouheylab/shared_datasets/point_tracking_data/ego_points")
    parser.add_argument("--seq_list", type=str, default="/scratch/sp7835/ego-tracking/evaluation_paths.json")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--model_weights",type=str, default="/vast/sp7835/ego-tracking/croco/pretrained_models/CroCo_V2_ViTBase_SmallDecoder.pth")
    args = parser.parse_args()
    main(args)
