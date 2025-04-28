import tarfile 
import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset
import pickle as pkl
import cv2
from torchvision import transforms
import pyrender
import sys
from pathlib import Path
import requests
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import trimesh
import random
import time
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from torch.utils.data import Sampler
from torch.utils.data import DataLoader
os.environ['DISPLAY'] = ':0'  # or any appropriate display like ':1' or ':99'
os.environ["PYOPENGL_PLATFORM"] = "egl"

from torch.utils.data import get_worker_info

def _get_worker_id():
    worker_info = get_worker_info()
    return worker_info.id if worker_info else "main"


DEPTH_ANYTHING_PATH = "/scratch/projects/fouheylab/dma9300/OSNOM/code/tracking_code/Depth-Anything"
class EpicKitchenDataset(Dataset):
    def __init__(self, root,  split = "train", overlap_threshold=0.20, cache_dir  = "cache_dir", train_ratio = 0.80):
        """
        Args:
            json_dir (str): Path to the directory containing JSON overlap files.
            overlap_threshold (float): Minimum overlap value to include a pair.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        
        is_training  = split == "train"
        self.transform = self.get_transform(train = is_training)
        self.pairs = []
        self.root  = root
        self.split = split
        self.overlap_dirs = os.path.join(self.root,"data", "sparse_overlaps")
        self.cameras = os.path.join(self.root, "data", "sparse_cameras")
        self.overlap_threshold = overlap_threshold
        self.depth_maps = os.path.join(self.root, "data", "depthmaps")
        self.tar_files  = os.path.join(self.root, "data", "preprocessed_data")
        self.tarfile_cache = {}   # cache {kitchen_id: TarFile}
        self.rescale_scores = json.load(open("/scratch/projects/fouheylab/dma9300/OSNOM/data/rescale_scores.json"))
        self.sparse_colmap_reconstruction  = "/scratch/projects/fouheylab/shared_datasets/epic-fields/Sparse/"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.metric_depth_model = None
        self.data = None
        self.unique_frames_per_video = {}
        self.train_ratio = train_ratio
        self._load_and_cache_all_tar_files()

        # self.backbone = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14", pretrained=True).cuda().eval()

       

        # Cached file paths
        train_cache = os.path.join(cache_dir, "train_pairs.pkl")
        val_cache = os.path.join(cache_dir, "val_pairs.pkl")


        if os.path.exists(train_cache) and os.path.exists(val_cache):
            if self.split == "train":
                with open(train_cache, 'rb') as f:
                    self.data = pkl.load(f)
            elif self.split == "val":
                with open(val_cache, 'rb') as f:
                    self.data = pkl.load(f)
        else:
            # Load all JSON files in the directory
            for filename in os.listdir(self.overlap_dirs):
                if filename.endswith("_overlaps.json"):
                    with open(os.path.join(self.overlap_dirs, filename), 'r') as f:
                        overlaps = json.load(f)
                        unique_frames  = set()
                        for img1, sub_dict in overlaps.items():
                            # breakpoint()
                            for img2, val in sub_dict.items():
                                # breakpoint()
                                if val["overlap"] > self.overlap_threshold and len(val["kp1"])>=15:
                                    self.pairs.append((img1, img2, filename.split("_overlaps")[0], val["overlap"], val["kp1"], val["kp2"]))
                                    unique_frames.add(img1)
                                    unique_frames.add(img2)
                        self.unique_frames_per_video[filename.split("_overlaps")[0]] = list(unique_frames)
            
            # if not os.path.exists(f'features_2d_dino_overlap_threshold_{self.overlap_threshold*100}_percent'):
            
            random.shuffle(self.pairs)
            split_idx = int(len(self.pairs) * self.train_ratio)
            train_pairs = self.pairs[:split_idx]
            test_pairs = self.pairs[split_idx:]

            # Save splits
            with open(train_cache, 'wb') as f:
                pkl.dump(train_pairs, f)
            with open(val_cache, 'wb') as f:
                pkl.dump(test_pairs, f)

            if self.split == "train":
                with open(train_cache, 'rb') as f:
                    self.data = pkl.load(f)
            elif self.split == "val":
                with open(val_cache, 'rb') as f:
                    self.data = pkl.load(f)

                                
        # self.setup_metric_depth_model(DEPTH_ANYTHING_PATH)
        # self.pre_compute_dino_features()
    

    def _load_and_cache_all_tar_files(self):
        """
        Pre-load and cache all .tar files per worker.
        """
        worker_id = _get_worker_id()
        for file_name in os.listdir(self.tar_files):
            if file_name.endswith(".tar"):
                kitchen_id = file_name.split(".")[0]
                key = f"{worker_id}_{kitchen_id}"
                if key not in self.tarfile_cache:
                    tar_path = os.path.join(self.tar_files, file_name)
                    self.tarfile_cache[key] = tarfile.open(tar_path, 'r')
    

    def get_transform(self, train=True):
        if train:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Add jitter here
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        return transform
    def read_image_from_tar(self, kitchen_id, file_name):
        """
        Reads an image from a pre-opened tar file.

        Parameters:
            kitchen_id (str): The kitchen ID corresponding to the tar file.
            file_name (str): The name of the image file inside the tar archive.

        Returns:
            numpy.ndarray: The image as a NumPy array.
        """
        # Get the tar file handle
        worker_id = _get_worker_id()
        key = f"{worker_id}_{kitchen_id}"

        # Load tar if not already cached (safe fallback)
        if key not in self.tarfile_cache:
            tar_path = os.path.join(self.tar_files, f"{kitchen_id}.tar")
            self.tarfile_cache[key] = tarfile.open(tar_path, 'r')

        tar = self.tarfile_cache[key]
        
        try:
            # Extract the file content from the tar file
            file_names = tar.getnames()
            # breakpoint()
            if f"./{file_name}" not in file_names:
                print(f"File {file_name} not found inside the tar archive.")
                return None

            file = tar.extractfile(f"./{file_name}")
            if file:
                # Read image bytes
                image_bytes = file.read()
                # Convert bytes to a NumPy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                # Decode the image using OpenCV
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                # breakpoint()
                if img is None:
                    print(f"Failed to decode the image {file_name}.")
                    return None
                return img
        except Exception as e:
            print(f"Error reading {file_name} from {kitchen_id}: {e}")
            return None


   
    def __len__(self):
        return len(self.data)

    def rescale_intrinsics(self, intrinsics, old_resolution, new_resolution):
        """
        Rescale the camera intrinsics for a new resolution.
        
        Parameters:
            intrinsics (numpy.array): The original camera intrinsics matrix (3x3).
            old_resolution (tuple): The original resolution (width, height).
            new_resolution (tuple): The new resolution (width, height).
        
        Returns:
            numpy.array: The rescaled intrinsics matrix.
        """
        # Unpack old and new resolution
        old_w, old_h = old_resolution
        new_w, new_h = new_resolution
        
        # Extract original intrinsic parameters
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # Focal lengths
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # Principal point
        
        # Rescale focal lengths and principal point
        fx_new = fx * new_w / old_w
        fy_new = fy * new_h / old_h
        cx_new = cx * new_w / old_w
        cy_new = cy * new_h / old_h
        
        # Create the new intrinsic matrix
        new_intrinsics = np.array([
            [fx_new, 0, cx_new],
            [0, fy_new, cy_new],
            [0, 0, 1]
        ])
        
        return new_intrinsics


    def rasterize_mesh_with_pyrender(self, mesh, K, RT, image_size):
        """
        Renders a depth map of the given mesh using specified camera intrinsics and extrinsics.

        Parameters:
        - mesh: The 3D mesh to render (trimesh.Trimesh).
        - K: (3, 3) numpy array representing the camera intrinsic matrix.
        - RT: (4, 4) numpy array representing the camera extrinsic matrix.
        - image_size: Tuple (width, height) specifying the dimensions of the output image.

        Returns:
        - depth: A 2D numpy array representing the depth map.
        """
        # Create pyrender mesh from trimesh
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

        # Create a new pyrender scene
        scene = pyrender.Scene()
        scene.add(pyrender_mesh)

        # Extract intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create camera
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

        # Adjust extrinsics to match pyrender's coordinate convention
        T = RT.copy()
        T[:, 1:3] *= -1  # COLMAP to computer graphics conversion

        # Add camera node
        scene.add(camera, pose=T)

        # Set up the offscreen renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=image_size[0], viewport_height=image_size[1])

        # Render depth
        _, depth = renderer.render(scene)

        # Clean up
        renderer.delete()

        # Post-process depth
        depth[depth <= 0] = np.inf

        return depth



    def download_checkpoint(self, url, save_path):
        """
        Downloads a file from a URL to a specified local path.
        """
        if os.path.exists(save_path):
            print(f'File {save_path} already exists.')
            return
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for data in tqdm(response.iter_content(1024 * 1024)):
                f.write(data)
    
    def setup_metric_depth_model(self, path_to_depth_anything):

        metric_depth_path = os.path.join(path_to_depth_anything, 'metric_depth')
        sys.path.append(metric_depth_path)  

        # Import necessary modules from ZoeDepth
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config


        # Ensure the checkpoints directory exists
        checkpoints_dir = Path(f'{path_to_depth_anything}/metric_depth/checkpoints')
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Define checkpoint paths and URLs
        checkpoints = {
            'depth_anything_vitl14': 'https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth',
            'depth_anything_metric_depth_indoor': 'https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt'
        }

        # Change directory for checkpoint downloads
        os.chdir(f'{path_to_depth_anything}/metric_depth')
        Path('checkpoints').mkdir(exist_ok=True)
        # Download checkpoints if they don't exist
        for name, url in checkpoints.items():
            self.download_checkpoint(url, checkpoints_dir / f'{name}.pth')

        # Load the ZoeDepth model for metric depth estimation
        config = get_config('zoedepth', 'eval', 'nyu')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config.pretrained_resource = f'local::{checkpoints_dir}/depth_anything_metric_depth_indoor.pt'
        self.metric_depth_model = build_model(config).to(device).eval()

    
    def estimate_metric_depth(self, image):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transform the input image to a tensor
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        # Perform inference to get the depth map
        with torch.no_grad():
            pred = self.metric_depth_model(image_tensor, dataset='nyu')

        # Extract the depth map from the model's output
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        depth = pred[0]

        # Resize the depth map to match the input image dimensions
        h, w = image.size[::-1]
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        return depth.detach().cpu().numpy()


    
    def align_depths(self, depth1, depth2, subsample_size = None):
        """
        Aligns depth1 to depth2 by estimating scale (alpha) and shift (beta) factors.

        Parameters:
        - depth1: numpy array of the first depth map (metric depth).
        - depth2: numpy array of the second depth map (rendered depth from mesh).
        - subsample_size: Optional integer specifying the number of random samples to use for estimation.

        Returns:
        - depth1_aligned: numpy array of depth1 aligned to depth2.
        - alpha: Estimated scale factor.
        - beta: Estimated shift factor.
        """
        # Flatten the depth maps
        depth1_flat = depth1.flatten()
        depth2_flat = depth2.flatten()

        # Subsample if required
        if subsample_size is not None:
            indices = np.random.choice(len(depth1_flat), subsample_size, replace=False)
            depth1_flat = depth1_flat[indices]
            depth2_flat = depth2_flat[indices]

        # Formulate the linear system
        A = np.vstack([depth1_flat, np.ones_like(depth1_flat)]).T
        solution, _, _, _ = np.linalg.lstsq(A, depth2_flat, rcond=None)
        alpha, beta = solution

        # Apply the transformation
        depth1_aligned = depth1 * alpha + beta

        return depth1_aligned

    def normalize_depth(self, depth):
        # Normalize depth for visualization (0-255)
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        if depth_max - depth_min > 0:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth)
        return (normalized * 255).astype(np.uint8)


    def compute_metric_depths(self, mesh, RT_c2w, intrinsic_matrix, image, kitchen_id, frame_name):
        
        save_dir = os.path.join(self.depth_maps, kitchen_id)
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Define the file path for the .npy file
        depth_file_path = os.path.join(save_dir, f'{frame_name.split(".")[0]}_aligned_depth.npy')
        
        if os.path.exists(depth_file_path):
            print(f"File {depth_file_path} already exists. Skipping saving.")
            return  # Exit the function or skip further operations

        # Render depth map using pyrender
        rendered_depth = self.rasterize_mesh_with_pyrender(
            mesh, intrinsic_matrix, RT_c2w, (224, 224)
        )

        # Estimate metric depth from the image
        metric_depth = self.estimate_metric_depth(image)

        # Align the metric depth with the rendered depth
        aligned_depth = self.align_depths(metric_depth, rendered_depth)

        np.save(depth_file_path, aligned_depth)
                




    def pre_computed_depths(self):
        '''
        This method processes each unique frame for every video in the dataset, performs necessary data loading,
        resizing, and mesh scaling, and computes metric depth maps for each image in the video. It is designed to
        handle multiple kitchen ID videos and frames within those videos, extracting the necessary 3D mesh and 
        camera data to compute depth maps based on the image and camera parameters.
        '''
        for kitchen_id, frames_names in self.unique_frames_per_video.items():

            for image_name in frames_names:

                mesh_path = os.path.join(self.root, "data", "aggregated", kitchen_id, "fused-minpix15-meshed-delaunay-qreg5.ply")
                mesh = trimesh.load(mesh_path, force='mesh')
                mesh.apply_scale(self.rescale_scores[kitchen_id])

                img  = self.read_image_from_tar(kitchen_id, image_name)

                # # # Resize images to 224x224
                img = cv2.resize(img, (224, 224))
                
                try:
                    camera_data = np.load(os.path.join(self.cameras, f'{kitchen_id}.npz'), allow_pickle=True)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                
                data = camera_data[image_name]
                RT_c2w = data.item()["RT"]
                intrinsics= self.rescale_intrinsics(data.item()["intrinsics"], (480, 256), (224, 224))

                self.compute_metric_depths(mesh, RT_c2w, intrinsics, img, kitchen_id, image_name)
    


    def compute_fundamental_matrix(self, K1, RT1_c2w, K2, RT2_c2w):
        """
        Computes the fundamental matrix F between two camera views given their
        intrinsic matrices and camera-to-world extrinsic matrices.

        Parameters:
        -----------
        K1 : np.ndarray of shape (3, 3)
            Intrinsic matrix of the first camera.
        RT1_c2w : np.ndarray of shape (4, 4)
            camera-to-world transformation matrix for the first camera.
        K2 : np.ndarray of shape (3, 3)
            Intrinsic matrix of the second camera.
        RT2_c2w: np.ndarray of shape (4, 4)
            camera-to-world transformation matrix for the second camera.

        Returns:
        --------
        F : np.ndarray of shape (3, 3)
            Fundamental matrix between the two camera views.
        """


        def skew(t):
            """Returns the skew-symmetric matrix of a 3D vector t."""
            return np.array([
                [0,      -t[2],  t[1]],
                [t[2],    0,    -t[0]],
                [-t[1],  t[0],    0]
            ])
            
        # Extract rotation and translation from camera-to-world matrices.
        R1, t1 = RT1_c2w[:3, :3], RT1_c2w[:3, 3]
        R2, t2 = RT2_c2w[:3, :3], RT2_c2w[:3, 3]

        R_rel = R2.T @ R1      # Relative rotation
        t_rel = R2.T @ (t1 - t2) # Relative translation

        # Compute the essential matrix from relative translation and rotation.
        E = skew(t_rel) @ R_rel

        # Compute the fundamental matrix.
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        return F
   

    # def compute_fundamental_matrix(self, K1, RT1_w2c, K2, RT2_w2c):
    #     """
    #     Computes the fundamental matrix F between two camera views given their
    #     intrinsic matrices and world-to-camera extrinsic matrices.

    #     Parameters:
    #     -----------
    #     K1 : np.ndarray of shape (3, 3)
    #         Intrinsic matrix of the first camera.
    #     RT1_w2c : np.ndarray of shape (4, 4)
    #         world-to-camera transformation matrix for the first camera.
    #     K2 : np.ndarray of shape (3, 3)
    #         Intrinsic matrix of the second camera.
    #     RT2_w2c: np.ndarray of shape (4, 4)
    #         world-to-camera transformation matrix for the second camera.

    #     Returns:
    #     --------
    #     F : np.ndarray of shape (3, 3)
    #         Fundamental matrix between the two camera views.
    #     """
    #     def skew(t):
    #         """Returns the skew-symmetric matrix of a 3D vector t."""
    #         return np.array([
    #             [0, -t[2], t[1]],
    #             [t[2], 0, -t[0]],
    #             [-t[1], t[0], 0]
    #         ])

    #     # Extract rotation and translation
    #     R1, t1 = RT1_w2c[:3, :3], RT1_w2c[:3, 3]
    #     R2, t2 = RT2_w2c[:3, :3], RT2_w2c[:3, 3]

    #     # Relative motion from cam1 to cam2
    #     R_rel = R2 @ R1.T
    #     t_rel = t2 - R_rel @ t1

    #     # Essential matrix
    #     E = skew(t_rel) @ R_rel

    #     # Fundamental matrix
    #     F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    #     return F


    def load_2d_dino_feature(self, kitchen_id, image_name, feature_dir="features_dino"):
        """
        Loads a saved DINO feature .npy file for a given kitchen ID and image name.

        Args:
            kitchen_id (str): Identifier of the kitchen/video.
            image_name (str): Name of the image/frame (without or with extension).
            feature_dir (str): Root directory where features are stored.

        Returns:
            np.ndarray: Loaded DINO feature array (N, C).
        """
    
        path = os.path.join( f'features_2d_dino_overlap_threshold_{self.overlap_threshold*100}_percent',kitchen_id,f'{image_name.split(".")[0]}.npy')

        if not os.path.exists(path):
            raise FileNotFoundError(f"DINO feature file not found: {path}")

        return np.load(path)
  

    def check_fundamental_matrix(self, pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99):
        """
        Compute the fundamental matrix given corresponding keypoints from two images.
        
        Args:
            pts1 (np.ndarray): Nx2 array of keypoint coordinates in image 1.
            pts2 (np.ndarray): Nx2 array of keypoint coordinates in image 2.
            method: Method for computing F; default uses RANSAC.
                    Options include cv2.FM_7POINT, cv2.FM_8POINT, cv2.FM_RANSAC, cv2.FM_LMEDS.
            ransacReprojThreshold (float): RANSAC inlier threshold.
            confidence (float): Confidence level for RANSAC.
            
        Returns:
            F (np.ndarray): 3x3 Fundamental matrix.
            mask (np.ndarray): Inlier mask (if using a robust method like RANSAC).
        """
    
        # Compute the fundamental matrix using OpenCV
        pts1 = np.asarray(pts1, dtype=np.float32)
        pts2 = np.asarray(pts2, dtype=np.float32)
        F, mask = cv2.findFundamentalMat(pts1, pts2, method,ransacReprojThreshold=1.0, confidence=0.99)
        
        return F, mask

    def normalize_F(self, F):
        """Normalize F using the Frobenius norm."""
        return F / np.linalg.norm(F, 'fro')

    def compute_epipolar_residuals(self, F, pts1, pts2):
        """
        Computes the absolute epipolar residuals:
            |x2^T F x1|
        for each keypoint correspondence.
        
        pts1 and pts2 should be of shape (N, 2) in pixel coordinates.
        """
        # Convert points to homogeneous coordinates.
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
        
        residuals = []
        for x1, x2 in zip(pts1_h, pts2_h):
            error = abs(np.dot(x2, np.dot(F, x1)))
            residuals.append(error)
        return np.array(residuals)

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
        kp1 = np.asarray(kp1)
        kp2 = np.asarray(kp2)

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
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

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



    def __getitem__(self, idx):
        # start = time.time()
        image_name1, image_name2, kitchen_id, overlap, kp1, kp2 = self.data[idx]
        view1_img  = self.read_image_from_tar(kitchen_id, image_name1)
        view2_img  = self.read_image_from_tar(kitchen_id, image_name2)

        # self.plot_and_save_keypoints_on_image_pair(view1_img, view2_img, kp1, kp2,  save_path=f'{image_name1}_{image_name2}.png')
        # breakpoint()
        try:
            camera_data = np.load(os.path.join(self.cameras, f'{kitchen_id}.npz'), allow_pickle=True)
        except Exception as e:
            print(f"Error reading {file}: {e}")
        
        data1 = camera_data[image_name1]
        RT1_w2c = np.linalg.inv(data1.item()["RT"])
        RT1_c2w = data1.item()["RT"]

        intrinsics1 = self.rescale_intrinsics(data1.item()["intrinsics"], (480, 256), (224, 224))

        data2 = camera_data[image_name2]
    
        RT2_w2c = np.linalg.inv(data2.item()["RT"])
        RT2_c2w = data2.item()["RT"]
        intrinsics2 = self.rescale_intrinsics(data2.item()["intrinsics"], (480, 256), (224, 224))

        if self.transform:
            view1_img = self.transform(view1_img)
            view2_img = self.transform(view2_img)

        # if np.random.rand() < 0.5 and self.split == "train":
        #     view1_img, kp1, view1_img, kp2 =  self.rotate_image_pair_and_keypoints(view1_img, kp1, view2_img, kp2)
    
        sample = {
            "img1": view1_img,
            "img2": view2_img,
            "RT1_w2c": torch.tensor(RT1_w2c, dtype=torch.float32),
            "RT2_w2c": torch.tensor(RT2_w2c, dtype=torch.float32),
            "intrinsics1": torch.tensor(intrinsics1, dtype=torch.float32),
            "intrinsics2": torch.tensor(intrinsics2, dtype=torch.float32),
            "kitchen": kitchen_id, 
            "overlap": torch.tensor(overlap, dtype=torch.float32),
            "kp1": torch.tensor(kp1, dtype=torch.float32),
            "kp2": torch.tensor(kp2, dtype=torch.float32),
        }
        return sample



class RandomSubsetSampler(Sampler):
    def __init__(self, data_source, num_samples, seed=None):
        self.data_source = data_source
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        n = len(self.data_source)
        if self.num_samples > n:
            raise ValueError(f"num_samples={self.num_samples} exceeds dataset size={n}")
        
        # Set seed for reproducibility if provided
        if self.seed is not None:
            random.seed(self.seed)
        
        indices = random.sample(range(n), self.num_samples)
        return iter(indices)

    def __len__(self):
        return self.num_samples


class ResampleableDataLoader:
    def __init__(self, dataset, batch_size, num_samples, sub_sample, **dataloader_kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dataloader_kwargs = dataloader_kwargs
        self.sub_sample = sub_sample
        self._create_loader()

    def _create_loader(self):
        if self.sub_sample:
            sampler = RandomSubsetSampler(self.dataset, self.num_samples)
        else:
            sampler  = None
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            **self.dataloader_kwargs
        )
        self.iterator = iter(self.loader)

    def resample(self):
        self._create_loader()

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        return next(self.iterator)

    def __len__(self):
        return len(self.loader)



if __name__ == "__main__":
    ROOT  = "/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/"
    # Initialize dataset
    dataset = EpicKitchenDataset(root = ROOT)
    # dataset.pre_computed_depths()
    # breakpoint()

    # Get a sample from the dataset
    sample = dataset[0]  # Fetch the first item from the dataset

    if sample:
        print("Sample loaded successfully:")
        print("Image 1 shape:", sample['img1'].shape if sample['img1'] is not None else "None")
        print("Image 2 shape:", sample['img2'].shape if sample['img2'] is not None else "None")
        print("Overlap:", sample['overlap'])
    else:
        print("Error: One of the images could not be loaded.")