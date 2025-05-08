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
