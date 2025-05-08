import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
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
from models import DinoDeltaModel, CrocoDeltaNet,CrocoF, DinoF,KeyPointNet
from utils.losses import * 
from utils import misc
from collections import defaultdict
import argparse
from torch.cuda.amp import autocast, GradScaler
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import torch
import torchvision.transforms as T
from PIL import Image

def visualize_dino_features(images, model, save_path="results.jpg", img_size=224, mask_threshold=0.6):
    """
    Visualize DINO-like features using PCA, optionally highlighting foreground regions.

    Args:
        images (List[PIL.Image]): List of PIL images to process.
        model (torch.nn.Module): Pretrained DINO-like model with `forward_features`.
        save_path (str): Path to save the output visualization.
        img_size (int): Image size (assumes square images).
        mask_threshold (float): Threshold to select foreground patches from PCA.
    """
    model.eval()
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size)),  
        T.CenterCrop(img_size),
        # T.Normalize([0.5], [0.5])  # adjust based on model's requirements
        T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    
    images_tensor = torch.stack([transform(img) for img in images])
    images_plot = ((images_tensor.cpu().numpy() * 0.5 + 0.5) * 255).transpose(0, 2, 3, 1).astype(np.uint8)

    with torch.no_grad():
        embeddings = model(images_tensor)
        x_norm_patchtokens = embeddings.cpu().numpy()
    
    # breakpoint()
    num_imgs = len(images)
    patch_size = img_size // 14
    flat_tokens = x_norm_patchtokens.reshape(num_imgs * patch_size * patch_size, -1)

    # Foreground mask by PCA
    pca_1d = PCA(n_components=1).fit_transform(flat_tokens)
    pca_1d = minmax_scale(pca_1d).reshape(num_imgs, patch_size * patch_size)
    masks = (pca_1d > mask_threshold)

    # 3D PCA for visualization
    pca_3d = PCA(n_components=3)
    fg_tokens = np.vstack([x_norm_patchtokens[i, masks[i], :] for i in range(num_imgs)])
    pca_features = minmax_scale(pca_3d.fit_transform(fg_tokens))

    # Split back into images
    split_indices = np.cumsum([np.sum(mask) for mask in masks])
    split_indices = np.insert(split_indices, 0, 0)
    pca_imgs = []
    for i in range(num_imgs):
        pca_result = np.zeros((patch_size * patch_size, 3), dtype='float32')
        pca_result[masks[i]] = pca_features[split_indices[i]:split_indices[i + 1]]
        pca_imgs.append(pca_result.reshape(patch_size, patch_size, 3))

    # Plot results
    fig, axs = plt.subplots(num_imgs, 2, figsize=(8, 4 * num_imgs))
    for i in range(num_imgs):
        axs[i, 0].imshow(images_plot[i])
        axs[i, 0].axis('off')
        axs[i, 1].imshow(pca_imgs[i])
        axs[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
    N, _ = keypoints.shape
    H, W = feat_map.shape[2], feat_map.shape[3]

    # Normalize keypoints to [-1, 1] for grid_sample
    norm_x = (keypoints[:, 0] / (W - 1)) * 2 - 1
    norm_y = (keypoints[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack((norm_x, norm_y), dim=1).view(1, N, 1, 2)  # (1, N, 1, 2)

    # Sample the features using bilinear interpolation
    sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # (1, C, N, 1)
    return sampled.squeeze(-1).squeeze(0).permute(1, 0)  # (N, C)



def find_nearest_neighbors(feat_px1, feat_px2, keypoints1, k=1):
    """
    Find k nearest neighbors using cosine similarity and return a visibility mask
    based on a percentile threshold of cosine distances.
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


from PIL import Image
import numpy as np

def load_valid_frames(root, valid_frames, target_size=(224, 224)):
    img_ls = []
    original_shape = None

    for valid_frame in valid_frames:
        img_path = f"{root}/frame_{str(valid_frame+1).zfill(10)}.jpg"
        img = Image.open(img_path)
        
        if original_shape is None:
            original_shape = img.size[::-1]  # (height, width)

        resized_img = img.resize(target_size, resample=Image.BILINEAR)
        img_ls.append(np.array(resized_img))

    return np.concatenate(img_ls, axis=1), original_shape


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


def load_image_pairs(root, valid_frames, target_size=(224, 224)):
    # Load target image (the first in the list)
    target_idx = valid_frames[0]
    target_path = f"{root}/frame_{str(target_idx + 1).zfill(10)}.jpg"
    target_img = Image.open(target_path).resize(target_size, Image.BILINEAR)

    # Generate image pairs (target, source)
    image_pairs = []
    for source_idx in valid_frames[0:]:
        source_path = f"{root}/frame_{str(source_idx + 1).zfill(10)}.jpg"
        source_img = Image.open(source_path)
        image_pairs.append((target_img.copy(), source_img))  # Use .copy() to avoid reference issues

    return image_pairs

def images_loader(root, valid_frames):
    image_pairs = []
    for source_idx in valid_frames:
        source_path = f"{root}/frame_{str(source_idx + 1).zfill(10)}.jpg"
        source_img = Image.open(source_path)
        image_pairs.append(source_img) 
    return image_pairs



def processed_pairs(image_pairs):
    processed_pairs = []
    for tgt_img, src_img in image_pairs:
        tgt_tensor = test_transform(np.array(tgt_img))
        src_tensor = test_transform(np.array(src_img))
        processed_pairs.append((tgt_tensor, src_tensor))
    return processed_pairs

def inference_loop(processed_pair, model, device="cpu"):
    with torch.no_grad():
        tgt_tensor, src_tensor = processed_pair
        tgt_tensor = tgt_tensor.unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)
        src_tensor = src_tensor.unsqueeze(0).to(device)
        feat_target, feat_source = model(tgt_tensor, src_tensor)
    return feat_target, feat_source
        


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_location", type=str, help="Path to EgoPoints folder", default="/scratch/projects/fouheylab/shared_datasets/point_tracking_data/ego_points")
parser.add_argument("--seq_name", type=str, help="EgoPoints sequence name to visualise", default="P08_21_start_21480_end_22217")
args = parser.parse_args()
print(f"Visualising tracks for {args.seq_name}...")

# Load annotations
annots = np.load(f"{args.dataset_location}/{args.seq_name}/annot.npz")
trajs_gt = annots["trajs_2d"]
valids = annots["valids"]
valid_frames = get_valid_frames(valids)



# Load concatenated frames (224x224 each)
concat_frames, img_shape = load_valid_frames(f"{args.dataset_location}/{args.seq_name}/rgbs", valid_frames, target_size=(224, 224))
trajs_gt = rescale_keypoints(trajs_gt,img_shape)


# Load and preprocess image pairs
image_pairs = load_image_pairs(f"{args.dataset_location}/{args.seq_name}/rgbs", valid_frames)
# breakpoint()
processed_pairs = processed_pairs(image_pairs)
images  = images_loader(f"{args.dataset_location}/{args.seq_name}/rgbs", valid_frames)
# model_vis = DinoF()
# visualize_dino_features(images, model_vis, save_path="results.jpg", img_size=224, mask_threshold=0.0)
# breakpoint()

# Initialize model
device = "cpu"
MODEL_WEIGHTS = "/scratch/projects/fouheylab/dma9300/OSNOM/croco_model_epochs_50_cosine_loss/model_loss_cosine_epoch_70_val_loss_0.0236.pth"

model = DinoF()
# checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
# model.load_state_dict(checkpoint, strict =False)
model.to(device)
model.eval()

# Set up plot
img_shape = (224, 224)
fig, axs = plt.subplots(1, 1, figsize=(8, 3))
axs.axis("off")
axs.imshow(concat_frames)
offset_x = img_shape[1]

plot_lines = {}
for frame_num, frame_idx in enumerate(valid_frames):
    processed_pair = processed_pairs[frame_num]

    for point_idx in range(valids.shape[1]):
        if valids[frame_idx, point_idx] != 1.0:
            continue

        if point_idx not in plot_lines:
            plot_lines[point_idx] = {"coords": []}

        # Initialize kpts1 only at the first frame
        if frame_num == 0:
            kpts1 = trajs_gt[frame_idx, point_idx]  # (2,)
        else:
            # Use the predicted keypoint from the previous frame
            kpts1 = kpts2[0][0].cpu().numpy()  # Assuming kpts2 is a list of tensors

        kpts1_tensor = torch.from_numpy(kpts1).to(torch.float32).unsqueeze(0)  # (1, 2)

        # Run model
        feat1, feat2 = inference_loop(processed_pair, model)

        # Track to next frame
        _, kpts2 = find_nearest_neighbors(feat1, feat2, kpts1_tensor, k=1)
        x = kpts2[0][0, 0].item()
        y = kpts2[0][0, 1].item()

        # Store tracked coordinates with image offset
        plot_lines[point_idx]["coords"].append([(x + (offset_x * frame_num)), y])

        # Optional: linestyle for dynamic/static objects
        if "dynamic_obj_tracks" in annots:
            is_dynamic = annots["dynamic_obj_tracks"][point_idx] == 1
            plot_lines[point_idx]["linestyle"] = ":" if is_dynamic else "-"
        else:
            plot_lines[point_idx]["linestyle"] = "-"

# Draw all tracked trajectories
for point_idx in plot_lines:
    coords = np.array(plot_lines[point_idx]["coords"])
    axs.plot(coords[:, 0], coords[:, 1], linestyle=plot_lines[point_idx]["linestyle"], marker="o")

fig.savefig(f"track_vis_{args.seq_name}_DINO_newwww.png", bbox_inches="tight")
plt.close(fig)