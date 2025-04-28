import os
import glob
import random
import math
import pickle as pkl
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset paths
DATA_PATH = "/scratch/projects/fouheylab/shared_datasets/point_tracking_data/"
PREPROCESSED_DATA_PATH = "/scratch/projects/fouheylab/dma9300/OSNOM/egopoints/"

class EgoPoints(Dataset):
    def __init__(self, root, split="train", max_frame_gap=8, num_seqs=4000):
        self.split = split
        self.root = root
        self.max_frame_gap = max_frame_gap
        self.num_seqs = num_seqs
        self.unique_kitchens = self.set_unique_kitchens()
        self.keys = []
        is_training = self.split == "train"
        self.transform = self.get_transform(train=is_training)

        # Generate or load data pairs
        base_path = f"{PREPROCESSED_DATA_PATH}frame_pairs_sample_size_{self.num_seqs}_max_frame_gap_{self.max_frame_gap}.pkl"
        if not os.path.exists(base_path):
            self.get_image_pairs()
        else:
            train_path = f"{PREPROCESSED_DATA_PATH}train_frame_pairs_sample_size_{self.num_seqs}_max_frame_gap_{self.max_frame_gap}.pkl"
            if not os.path.exists(train_path):
                self.split_data()

        # Load dataset split
        data_path = f"{PREPROCESSED_DATA_PATH}{self.split}_frame_pairs_sample_size_{self.num_seqs}_max_frame_gap_{self.max_frame_gap}.pkl"
        with open(data_path, "rb") as f:
            self.pairs_dict = pkl.load(f)

        self.keys = list(self.pairs_dict.keys())

    def set_unique_kitchens(self):
        """Randomly select a subset of kitchens for this dataset."""
        all_kitchens = os.listdir(os.path.join(self.root, "k_epic", "train"))
        return random.sample(all_kitchens, min(self.num_seqs, len(all_kitchens)))

    def split_data(self):
        """Split the full dataset into training and validation sets."""
        print("Creating dataset splits...")
        base_path = f"{PREPROCESSED_DATA_PATH}frame_pairs_sample_size_{self.num_seqs}_max_frame_gap_{self.max_frame_gap}.pkl"
        with open(base_path, "rb") as f:
            pairs_dict = pkl.load(f)

        all_keys = list(pairs_dict.keys())
        random.shuffle(all_keys)
        split_index = int(0.8 * len(all_keys))

        train_pairs = {k: pairs_dict[k] for k in all_keys[:split_index]}
        val_pairs = {k: pairs_dict[k] for k in all_keys[split_index:]}

        with open(f"{PREPROCESSED_DATA_PATH}train_frame_pairs_sample_size_{self.num_seqs}_max_frame_gap_{self.max_frame_gap}.pkl", "wb") as f:
            pkl.dump(train_pairs, f)
        with open(f"{PREPROCESSED_DATA_PATH}val_frame_pairs_sample_size_{self.num_seqs}_max_frame_gap_{self.max_frame_gap}.pkl", "wb") as f:
            pkl.dump(val_pairs, f)


    def rotate_image_pair_and_keypoints(self, image_tensor1, keypoints1, image_tensor2, keypoints2, center=None, angle_range=(-45, 45)):
        """
        Deterministically rotate two image tensors and their corresponding numpy keypoints with the same angle.
        Returns only keypoints that remain in bounds in both images.
        """

        keypoints1 = np.asarray(keypoints1, dtype=np.float32)
        keypoints2 = np.asarray(keypoints2, dtype=np.float32)

        angle_deg = random.uniform(*angle_range)

        # Assume image tensors have shape (C, H, W)
        _, h, w = image_tensor1.shape
        if center is None:
            center = (w * 0.5, h * 0.5)

        # Use torchvision functional to rotate images
        img1_rot = TF.rotate(image_tensor1, angle=angle_deg, center=center)
        img2_rot = TF.rotate(image_tensor2, angle=angle_deg, center=center)

        # Build the affine matrix used for keypoints
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        cx, cy = center

        M = np.array([
            [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
            [sin_a,  cos_a, cy - sin_a * cx - cos_a * cy]
        ])  # Shape: (2, 3)

        # Apply affine transform to keypoints
        kps1_h = np.hstack([keypoints1, np.ones((keypoints1.shape[0], 1))])  # (N, 3)
        kps2_h = np.hstack([keypoints2, np.ones((keypoints2.shape[0], 1))])
        kps1_rot = (M @ kps1_h.T).T  # (N, 2)
        kps2_rot = (M @ kps2_h.T).T

        # In-bounds check
        mask1 = (
            (kps1_rot[:, 0] >= 0) & (kps1_rot[:, 0] < w) &
            (kps1_rot[:, 1] >= 0) & (kps1_rot[:, 1] < h)
        )
        mask2 = (
            (kps2_rot[:, 0] >= 0) & (kps2_rot[:, 0] < w) &
            (kps2_rot[:, 1] >= 0) & (kps2_rot[:, 1] < h)
        )

        mask = mask1 & mask2

        return img1_rot, kps1_rot[mask], img2_rot, kps2_rot[mask]

    def get_image_pairs(self):
        """Generate frame pairs with annotations for all selected kitchens."""
        print("Generating data pairs...")
        pairs_dict = {}

        for kitchen in tqdm(self.unique_kitchens):
            rgb_dir = os.path.join(self.root, "k_epic", "train", kitchen, "rgbs")
            rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
            annotations = np.load(os.path.join(self.root, "k_epic", "train", kitchen, "annot.npz"))

            for i in range(len(rgb_paths) - 1):
                for j in range(i + 1, len(rgb_paths) - 1, self.max_frame_gap):
                    key = f"{kitchen}_{frame1_path.split('/')[-1].split('.')[0]}_{frame2_path.split('/')[-1].split('.')[0]}"
                    pairs_dict[key] = {
                        'frame1_path': rgb_paths[i],
                        'frame2_path': rgb_paths[j],
                        'frame1_trajs_2d': annotations["trajs_2d"][i],
                        'frame2_trajs_2d': annotations["trajs_2d"][j],
                        'frame1_visibs': annotations["visibs"][i],
                        'frame2_visibs': annotations["visibs"][j],
                        'frame1_valids': annotations["valids"][i],
                        'frame2_valids': annotations["valids"][j],
                    }

        # Save all, train, and val
        with open(f"{PREPROCESSED_DATA_PATH}frame_pairs_sample_size_{self.num_seqs}.pkl", "wb") as f:
            pkl.dump(pairs_dict, f)
        self.split_data()

    def get_transform(self, train=True):
        """Image preprocessing and augmentation."""
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224))
        ]
        if train:
            transform_list.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.keys)

    def load_image(self, path):
        """Load image and convert BGR to RGB."""
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_image_pair(self, path1, path2):
        return self.load_image(path1), self.load_image(path2)

    def resize_points(self, pts, img, new_size=(224, 224)):
        """Resize keypoints to match resized image dimensions."""
        h, w = img.shape[:2]
        scale_x, scale_y = new_size[1] / w, new_size[0] / h
        pts = pts.copy()
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        return pts

    def plot_and_save_keypoints_on_image_pair(self, view1_img, view2_img, kp1, kp2, save_path='keypoint_match.png'):
        """
        Plot keypoints on a pair of images side by side and save the result.

        Args:
            view1_img (np.ndarray): First image (H, W, 3).
            view2_img (np.ndarray): Second image (H, W, 3).
            kp1 (np.ndarray): Keypoints in the first image (N, 2).
            kp2 (np.ndarray): Keypoints in the second image (N, 2).
            image_name1 (str): Name of the first image (optional).
            image_name2 (str): Name of the second image (optional).
            save_path (str): Path to save the plotted image (should end in .png).
        """
        # Ensure keypoints are numpy arrays
        kp1 = np.asarray(kp1)[0:20,:]
        kp2 = np.asarray(kp2)[0:20,:]

        view1_img = cv2.resize(view1_img, (224,224), interpolation=cv2.INTER_LINEAR)
        view2_img = cv2.resize(view2_img, (224,224), interpolation=cv2.INTER_LINEAR)

        # Create a side-by-side image
        h1, w1, _ = view1_img.shape
        h2, w2, _ = view2_img.shape
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1, :] = view1_img
        canvas[:h2, w1:, :] = view2_img

        # Shift kp2 by the width of the first image
        kp2_shifted = kp2.copy()
        kp2_shifted[:, 0] += w1

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(canvas)
        for p1, p2 in zip(kp1, kp2_shifted):
            color = np.random.rand(3,)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1)
            ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color=color, s=10)

        ax.axis('off')
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def set_epoch(self, epoch, subset_size=10000):
        """Resample self.keys deterministically based on the epoch."""
        rng = np.random.default_rng(seed=epoch + 777)
        all_keys = list(self.pairs_dict.keys())
        permuted_keys = rng.permutation(all_keys)

        # Repeat and truncate to get exactly `subset_size` samples
        repeated_keys = np.tile(permuted_keys, (1 + subset_size // len(permuted_keys)))[:subset_size]
        # breakpoint()
        self.keys = list(repeated_keys)
        assert len(self.keys) == subset_size

    def __getitem__(self, idx):
        key = self.keys[idx]
        pair = self.pairs_dict[key]
        try:
            # Load and resize images and keypoints
            img1, img2 = self.load_image_pair(pair["frame1_path"], pair["frame2_path"])
            kp1 = self.resize_points(pair["frame1_trajs_2d"], img1)
            kp2 = self.resize_points(pair["frame2_trajs_2d"], img2)

            vis1, vis2 = pair["frame1_visibs"], pair["frame2_visibs"]
            val1, val2 = pair["frame1_valids"], pair["frame2_valids"]

            # Only keep valid, visible keypoints in both frames
            mask = vis1.astype(bool) & vis2.astype(bool) & val1.astype(bool) & val2.astype(bool)
            kp1 = kp1[mask, :]
            kp2 = kp2[mask, :]

            # self.plot_and_save_keypoints_on_image_pair(img1, img2, kp1, kp2,  save_path=f'im1_im2_egopoints.png')

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

        
            return {
                "img1": img1,
                "img2": img2,
                "kp1": torch.tensor(kp1, dtype=torch.float32),
                "kp2": torch.tensor(kp2, dtype=torch.float32),
            }
        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
            return None




def main():
    dataset = EgoPoints(root=DATA_PATH, split="train")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]  # Trigger __getitem__

if __name__ == "__main__":
    main()
