import sys
sys.path.append('./code/tracking_code')
import torch.nn as nn
from util import *
import pickle
import pyrender
import trimesh
import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import cv2
import json
import requests
import tarfile
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

rescale_scores = json.load(open(os.path.join(os.path.dirname(__file__), '../../../data/rescale_scores.json')))
boxes_scales = {
    "P24_05": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P03_04": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P01_14": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P30_107": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P05_08": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P12_101": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P28_103": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P10_04": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P30_05": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P06_101": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P04_05": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P06_103": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P35_109": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P37_103": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P04_11": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P04_21": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P04_109": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P02_07": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P28_14": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P22_01": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P15_02": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P04_26": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P01_09": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P02_109": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P02_101": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P24_08": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P23_05": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P28_110": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P20_03": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P11_105": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P08_09": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P22_07": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P03_113": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P04_02": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P25_107": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P02_130": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P08_16": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P30_101": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P18_07": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P01_103": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P01_05": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P03_03": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P11_102": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P06_107": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P37_101": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P06_12": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P02_107": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P06_13": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P28_109": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P12_02": {"x_scale": 0.3563, "y_scale": 0.3556},
    "P12_03": {"x_scale": 0.3563, "y_scale": 0.3556},
    "P04_25": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P08_21": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P02_128": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P04_03": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P14_05": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P23_02": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P28_112": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P06_01": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P07_08": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P11_103": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P02_132": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P06_14": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P02_01": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P18_03": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P06_102": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P01_01": {"x_scale": 0.2375, "y_scale": 0.2370},
    "P35_105": {"x_scale": 0.2375, "y_scale": 0.2370}
}


import torchvision.transforms as transforms
from torchvision.transforms import Compose
import torch.nn.functional as F


def get_depth_anything(image, depth_anything_model, transform, device='cuda'):
    image = image / 255.0

    h, w = image.shape[:2]

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = depth_anything_model(image)

    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    return depth

def get_depth_anything_metric(image, model, device='cuda'):
    image_pil = transforms.ToPILImage()(image)
    image_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image_tensor, dataset='nyu')

    if isinstance(pred, dict):
        pred = pred.get('metric_depth', pred.get('out'))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    depth = pred[0]

    h, w = image.shape[:2]
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    return depth


def draw_random_semantic_mask(ann):
    ann = np.asarray(ann, dtype=np.int32)  # Ensure the annotation is an integer numpy array

    # Create an empty image with the same dimensions as the annotation mask
    mask = np.zeros((ann.shape[0], ann.shape[1], 3), dtype=np.uint8)

    # Generate a random color for each unique value in the annotation mask, excluding the background (assumed to be 0)
    unique_classes = np.unique(ann)
    colors = {cls: np.random.randint(0, 256, size=3, dtype=np.uint8) for cls in unique_classes if cls != 0}

    # Fill the mask with random colors according to the annotation mask
    for cls, color in colors.items():
        mask[ann == cls] = color

    return mask


def align_depth1_to_depth2(depth1, depth2, mask, subsample_size=None):
    # depth1, depth
    depth1_masked = depth1[mask].cpu()
    depth2_masked = depth2[mask]
    if subsample_size is not None:
        subsample = np.random.randint(0, mask.sum(), subsample_size)
        depth1_masked = depth1_masked[subsample]
        depth2_masked = depth2_masked[subsample]

    # Solve for depth_masked * alpha + beta = rend_depth_masked
    A = np.stack([depth1_masked, np.ones_like(depth1_masked)], axis=-1)
    # A = torch.stack([depth1_masked, torch.ones_like(depth1_masked)], dim=-1)

    alpha, beta = np.linalg.lstsq(A, depth2_masked, rcond=None)[0]

    depth1_aligned = depth1 * alpha + beta
    print(f'alpha: {alpha}, beta: {beta}')

    return depth1_aligned.cpu().numpy()
def rename_keys(kitchen, original_dict, mapping_dict):
    # Create a new dictionary with the keys renamed
    renamed_dict = {}
    for old_key, sub_dict in original_dict.items():
        try:
            new_key = mapping_dict[kitchen][
                kitchen + '_' + old_key + '.jpg']
            renamed_dict[new_key.split('.')[0]] = sub_dict
        except:
            renamed_dict[old_key] = sub_dict

    return renamed_dict

class PHALP(nn.Module):
    """
    PHALP class is responsible for processing 3D object feature extraction using depth estimation.
    It handles loading and running models for depth prediction and feature extraction.

    Attributes:
        output_dir (str): Directory to save extracted features.
        data_path (str): Path to dataset containing poses, masks, and annotations.
        mesh_path (str): Path to 3D mesh data.
        frames_path (str): Directory for video frames.
        kitchen (str): Identifier for the specific kitchen or dataset instance.
    """
    def __init__(self, output_dir, data_path, mesh_path, frames_path, kitchen):
        """
        Initializes the PHALP class, loads models for depth estimation, and prepares paths for data processing.

        Args:
            output_dir (str): Directory to save extracted features.
            data_path (str): Path to dataset containing poses, masks, and annotations.
            mesh_path (str): Path to 3D mesh data.
            frames_path (str): Directory for video frames.
            kitchen (str): Identifier for the specific kitchen or dataset instance.
        """
        super(PHALP, self).__init__()

        # Add Depth-Anything path to system path
        path_to_depth_anything = '/scratch/projects/fouheylab/dma9300/OSNOM/code/tracking_code/Depth-Anything/'
        sys.path.append(path_to_depth_anything)

        # Import Depth-Anything model and utilities
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        ENCODER = 'vitl'

        # Load Depth-Anything model
        currdir = os.getcwd()
        os.chdir(f'{path_to_depth_anything}/')
        depth_anything = DepthAnything.from_pretrained(
            f'LiheYoung/depth_anything_{ENCODER}14'
        ).to(DEVICE).eval()
        os.chdir(currdir)

        # Define image transformation for Depth-Anything
        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # Lambda function for depth estimation
        self.image_to_depth = lambda image: get_depth_anything(
            image, depth_anything, transform, device=DEVICE
        )

        # Import Zoedepth model for metric depth estimation
        sys.path.append(f'{path_to_depth_anything}/metric_depth')
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config

        # Function to download checkpoint files if they don't exist
        def download_to_path(url, path):
            if os.path.exists(path):
                print(f'File {path} already exists')
                return
            response = requests.get(url, stream=True)
            with open(path, 'wb') as f:
                for data in tqdm(response.iter_content(1024 * 1024)):
                    f.write(data)

        # Change directory for checkpoint downloads
        os.chdir(f'{path_to_depth_anything}/metric_depth')
        Path('checkpoints').mkdir(exist_ok=True)

        # Download pre-trained models
        download_to_path(
            'https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth',
            'checkpoints/depth_anything_vitl14.pth'
        )
        download_to_path(
            'https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt',
            'checkpoints/depth_anything_metric_depth_indoor.pt'
        )

        # Load Zoedepth model for metric depth
        config = get_config('zoedepth', "eval", 'nyu')
        config.pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
        model_metric = build_model(config).to(DEVICE).eval()
        os.chdir(currdir)

        # Lambda function for metric depth estimation
        self.image_to_depth_metric = lambda image: get_depth_anything_metric(
            image, model_metric, device=DEVICE
        )

        # Initialize paths and directories
        self.RGB_tuples = get_colors()
        self.path_to_save = output_dir
        self.kitchen = kitchen
        self.mesh_path = mesh_path
        self.output_dir_name = f"saved_feat_3D/{self.kitchen}"
        self.path_to_save = os.path.join(self.path_to_save, self.output_dir_name)

        # Create output directory if it doesn't exist
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)

        self.data_path = data_path
        self.frames_path = frames_path

        # Load pose data
        with open(os.path.join(self.data_path, 'poses.json'), 'r') as f:
            self.poses = json.load(f)

        # Read other necessary data
        self.masks, _, self.camera_poses, self.frames, _ = read_data_1(self.data_path, 
                                                                    '', kitchen, True)

        # Load frame mapping data
        with open('./data/dense_frame_mapping_corrected.json') as f:
            self.mapping_dense = json.load(f)

        # Process bounding boxes and segmentations
        self.bbs_dict = get_object_bbs_seg(self.masks['video_annotations'])
        self.bbs_dict = rename_keys(self.kitchen, self.bbs_dict, self.mapping_dense)

        # Set device for computation
        self.device = DEVICE
        print('Device: ', self.device)

        # Set model to evaluation mode
        self.eval()
    

    def read_image_from_tar(self, tar_file_path, file_name):
        """
        Reads an image file from a .tar archive and converts it to OpenCV format (NumPy array).
        
        Parameters:
            tar_file_path (str): Path to the .tar archive.
            file_name (str): Name of the image file inside the tar archive to read.
        
        Returns:
            numpy.ndarray: The image as a NumPy array in OpenCV format.
        """
        ### gimage
            #  .tar -> images 
            #  .tar -> folder->images

        # with tarfile.open(tar_file_path, "r") as tar:
        #     file_names = tar.getnames()  # List all files inside the tar
        #     breakpoint()

        # breakpoint()
        try:
            # Open the tar file in read mode
            with tarfile.open(tar_file_path, "r") as tar:
                # Try to extract the specific image file
                file = tar.extractfile(f"./{file_name}")
             
                
                if file:
                    # Read the image content as bytes
                    image_bytes = file.read()
                    
                    # Convert the bytes to a NumPy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    
                    # Decode the NumPy array into an OpenCV image
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Check if the image was successfully decoded
                    if img is None:
                        print(f"Failed to decode the image {file_name}.")
                        return None
                    return img
                else:
                    print(f"File {file_name} not found inside the tar archive.")
                    return None
        except KeyError:
            print(f"The file {file_name} does not exist inside the tar archive.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
   

    def rescale_all_bounding_boxes(self, bboxes, x_scale, y_scale):
        """
        Rescales all bounding boxes in the list according to the given x and y scale factors.

        Parameters:
        bboxes (list of tuples/lists): List of bounding boxes, where each bounding box is represented 
                                        as (x_min, y_min, x_max, y_max).
        x_scale (float): Scale factor for the x-coordinates.
        y_scale (float): Scale factor for the y-coordinates.

        Returns:
        list of tuples: List of rescaled bounding boxes.
        """
        rescaled_bboxes = []
        
        # Loop through each bounding box and rescale it
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min_rescaled = x_min * x_scale
            y_min_rescaled = y_min * y_scale
            x_max_rescaled = x_max * x_scale
            y_max_rescaled = y_max * y_scale
            rescaled_bboxes.append((x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled))

        return rescaled_bboxes

    def track(self):
        """
        Performs the 3D feature extraction and depth estimation for each frame in the video.

        The method extracts bounding boxes and 3D features for objects in the frames,
        aligns the depth estimates, and saves the resulting 3D features in a pickle file.

        Returns:
            None
        """
        save_dict = {}  # Dictionary to store extracted 3D features

        # Load the 3D mesh for the kitchen
        mesh_path = os.path.join(self.mesh_path, 'fused-minpix15-meshed-delaunay-qreg5.ply')
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_scale(rescale_scores[self.kitchen])

        # Camera parameters: size and focal length
        image_size = [self.poses['camera']['width'], self.poses['camera']['height']]
        focal_length = self.poses['camera']['params'][:4]
        camera_intrinsics = [image_size[0], image_size[1]] + focal_length

        # Create Pyrender camera with the appropriate field of view
        camera = pyrender.PerspectiveCamera(yfov=2 * np.arctan(image_size[1] / (2 * focal_length[1])))
        SCENE = pyrender.Scene()
        RENDERER = pyrender.OffscreenRenderer(image_size[0], image_size[1])

        # Add mesh and camera to the scene
        pmesh = pyrender.Mesh.from_trimesh(mesh)
        SCENE.add(pmesh)
        CAMERA_NODE = SCENE.add(camera, pose=np.eye(4))

        batched_bbs = []  # List to store bounding boxes from all frames
        frame_names = []  # List to store names of frames
        res = (854, 480)  # Resolution for depth maps

        # Iterate over frames and process each
        for t_, frame_name in tqdm(enumerate(sorted(self.frames)), total=len(self.frames)):
            segments, bbs, objs = self.bbs_dict[f"{self.kitchen}_{frame_name}"]
            if len(bbs) == 0:
                continue  # Skip frames with no objects detected

            # Get camera pose for the current frame
            camera_pose = get_camera_pose_1(self.camera_poses, frame_name)
            depth = get_depth_shared(camera_pose, SCENE, CAMERA_NODE, RENDERER)
            frame_path = os.path.join(self.frames_path, f"{frame_name}.jpg")

            # Estimate depth for the frame
            try:
                # Read and preprocess the image frame
                image_frame = self.read_image_from_tar(self.frames_path, f"{frame_name}.jpg")
                image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
            except:
                continue  # Skip frames that cannot be loaded

            try:
                rend_depth = depth
            except FileNotFoundError:
                continue  # Skip frames where depth is not found

            # Create segmentation masks for detected objects
            height, width = depth.shape[1], depth.shape[0]
            mask_frames = []
            for s in segments:
                mask = np.zeros((480, 854), dtype=np.uint8)
                cv2.fillPoly(mask, np.int32([s]), color=1)
                resized_mask = cv2.resize(mask, (456, 256), interpolation=cv2.INTER_NEAREST)
                out_image = np.stack((resized_mask * 255,) * 3, axis=-1)
                mask_frames.append(out_image)

            if mask_frames:
                segs_all = np.stack(mask_frames, axis=0)
                segs_final = (segs_all > 0).any(axis=-1).any(axis=0)

            # Estimate depth metric and align with rendered depth
            depth_metric = self.image_to_depth_metric(image)
            if mask_frames:
                mask = (~segs_final) & (rend_depth > 1e-6)
            else:
                mask = rend_depth > 1e-6

            depth_metric_aligned = align_depth1_to_depth2(depth_metric, rend_depth, mask, None)

            if bbs:
                bbs  = self.rescale_all_bounding_boxes(bbs,boxes_scales[self.kitchen]["x_scale"], boxes_scales[self.kitchen]["y_scale"] )
                loca_features, loc_3d_objs, r3d_objs = extract_3d_features(
                    bbs, objs, camera_pose, depth_metric_aligned, camera_intrinsics,
                    self.data_path, frame_name, res
                )
                loca_features = loca_features.squeeze(1)
                features_3d = loca_features[:, :3]
                save_dict[frame_name] = (features_3d.cpu().numpy(), r3d_objs.cpu().numpy(), objs)

        # Save the extracted 3D features to disk
        with open(os.path.join(self.path_to_save, f'3D_feat_{self.kitchen}.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)


def main():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--output_dir", required=True, help="Output directory path")
    parser.add_argument("--data_path", required=True, help="Data path")
    parser.add_argument("--mesh_path", required=True, help="Data path")
    parser.add_argument("--frames_path", required=True, help="Frames path")
    parser.add_argument("--kitchen", required=True, help="Frames path")

    args = parser.parse_args()

    # Initialize your class with the configuration file argument
    phalp_instance = PHALP(args.output_dir, args.data_path, args.mesh_path, args.frames_path,
                           args.kitchen)
    phalp_instance.track()


if __name__ == "__main__":
    main()
