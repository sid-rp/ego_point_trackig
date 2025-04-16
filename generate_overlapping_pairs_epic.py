import numpy as np
import tarfile
import cv2
import numpy as np
import pickle as pkl
import os
import argparse
import json 
import trimesh
import random
from read_write_model import read_images_binary  # from COLMAP's scripts
import read_write_model


rescale_scores = json.load(open("/scratch/projects/fouheylab/dma9300/OSNOM/data/rescale_scores.json"))

MESH_ROOT_PATH = "/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/aggregated/"



def sample_points_in_camera_frustrum(image_dims, max_depth, sample_size, K, RT_w2c):
        """
        Randomly samples n points in the pixel space of the camera, assigns random depths, and converts to world coordinates.
        
        Parameters:
        - sample_size (int): Number of points to sample.
        - image_dims (tuple): Dimensions of the image (width, height,3).
        - depth_range (tuple): Minimum and maximum depth values as (min_depth, max_depth).
        - K (np.array): Camera intrinsic matrix (3x3).
        - RT_w2c(np.array): Camera rotation-translation matrix (3x4)
        Returns:
        - world_points (np.array): Array of sampled points in world coordinates (n x 3).
        """

        height, width = image_dims
        x_coords  = np.linspace(0,1, sample_size) * width
        y_coords = np.linspace(0,1, sample_size)* height
        
        # Stratified depth sampling 
        depths = np.linspace(0.1, max_depth, sample_size)
        
        # Create homogeneous 2D points in pixel space
        pixel_coords = np.vstack((x_coords, y_coords, np.ones(sample_size)))
        
        K_inv = np.linalg.inv(K)

        points_camworld = K_inv @ pixel_coords

        points_camworld = points_camworld * depths[None, :]  ## 3 x N

        points_camworld_hom  = np.vstack((points_camworld, np.ones(sample_size)))
        
        # Transform to world coordinates using the inverse of RT
        RT_c2w = np.linalg.inv(RT_w2c)
    
        points_world = RT_c2w @ points_camworld_hom

        points_world  = points_world[0:3, :]
 
        return points_world

def project_world_point2_image_pixels(world_points, RT_w2c, K):
        """
        Projects 3D points from the world coordinate frame to 2D image pixel coordinates.

        Args:
            world_points (numpy.ndarray): A NumPy array of shape (N, 3), representing N 3D points in the world coordinate frame.
            RT_w2c (numpy.ndarray): A 4x4 transformation matrix representing the rotation and translation 
                                from the world coordinate frame to the camera coordinate frame.
            K (numpy.ndarray): A 3x3 camera intrinsic matrix.

        Returns:
            numpy.ndarray: A NumPy array of shape (3, N) containing the 2D pixel coordinates (x, y) 
                        and the depth (z) for each projected point.
                        - The first row contains the x-coordinates (pixels).
                        - The second row contains the y-coordinates (pixels).
                        - The third row contains the z-depth values.
        """
        
        # convert world_points to homogeneous coordinates. Shape: (N, 4)
        points_world_hom = np.hstack((world_points, np.ones((world_points.shape[0], 1))))

        # Transform points from the world coordinate frame to the camera coordinate frame
        points_cam = np.matmul(RT_w2c, points_world_hom.T)

        # Extract the first three rows (x, y, z) from the transformed points. Shape: (3, N)
        points_cam = points_cam[0:3, :]

        # Extract the z-coordinates for depth normalization. Shape: (1, N)
        z = points_cam[2:3, :]

        # Apply the intrinsic matrix K to project 3D points to the image plane. Shape: (3, N)
        pixel_points = np.matmul(K, points_cam)

        # normalize xy by z
        xy = pixel_points[:2, :] / z

        # Concatenate x, y pixel coordinates with the z-depth values. Shape: (3, N)
        xyz = np.concatenate([xy, z])

        return xyz


def check_points_within_in_second_camera_frustrum(image_dim, world_points, RT_w2c, K, max_depth):
        """
        Checks the visibility of 3D world points in the second camera by projecting them 
        into the image plane and applying validity constraints.

        Parameters:
        -----------
        image_dim : np.ndarray
            The RGB image from the second camera, used to determine valid image bounds.
        world_points : np.ndarray, shape (N, 3)
            The 3D points in the world coordinate frame to be projected.
        RT_w2c : np.ndarray, shape (4, 4)
            The 4x4 transformation matrix (rotation + translation) from world coordinates 
            to the second camera's coordinate frame.
        K : np.ndarray, shape (3, 3)
            The intrinsic matrix of the second camera.
        max_depth : float
            The maximum depth threshold to consider a point valid.

        Returns:
        --------
        valid_mask : np.ndarray, shape (N,)
            A boolean mask indicating which world points are visible in the second camera.
            A point is considered visible if:
            - It projects within the image bounds.
            - Its depth in the camera coordinate frame is positive and less than or equal to `max_depth`.
        """

        # Project world points to camera coordinate frame
        points_hom = np.hstack((world_points.T, np.ones((world_points.T.shape[0], 1))))

        # Transform points from the world coordinate frame to the camera coordinate frame
        points_cam = np.matmul(RT_w2c, points_hom.T)
            
        # transform world points to pixels in the image plane (x/z, y/z, z)
        img_pts =  project_world_point2_image_pixels(world_points.T, RT_w2c, K)

        valid_mask = ((np.round(img_pts[0,:]).astype(np.int64)>=0) & (np.round(img_pts[0, :]).astype(np.int64) < image_dim[1])
                        & (np.round(img_pts[1,:].astype(np.int64))>=0)& (np.round(img_pts[1, :]).astype(np.int64) <  image_dim[0]) 
                        & (points_cam[2, :] <= max_depth)
                        & (points_cam[2, :] > 0.00))

        return valid_mask

def compute_intersection_between_2_camera_frustrums( image_dim, RT1_w2c, K1, RT2_w2c, K2, max_depth):
        """
        Computes the Intersection over Union (IoU) between the frustums of two cameras.

        Args:
            rgb_image (ndarray): The RGB image used to infer the shape of the camera frustum.
            RT1_w2c (ndarray): The 4x4 extrinsic matrix (rotation and translation) in of the first camera.
            K1 (ndarray): The 3x3 intrinsic matrix of the first camera.
            RT2_w2c (ndarray): The 4x4 extrinsic matrix (rotation and translation) of the second camera.
            K2 (ndarray): The 3x3 intrinsic matrix of the second camera.
            max_depth (float): The maximum depth value up to which points are sampled in the camera frustums.

        Returns:
            float: The IoU value representing the overlap between the two camera frustums.
        """
        # sample points evenly in the frustrums of the two cameras
        world_points_camera1 = sample_points_in_camera_frustrum(image_dim, max_depth, sample_size=10000, K=K1, RT_w2c=RT1_w2c)
        world_points_camera2 = sample_points_in_camera_frustrum(image_dim, max_depth, sample_size=10000, K=K2, RT_w2c=RT2_w2c)

        # check for points in one camera which are visible in the other camera
        camera1_pts_visible_camera2 = check_points_within_in_second_camera_frustrum(image_dim, world_points_camera1, RT2_w2c, K2, max_depth)
        camera2_pts_visible_camera1 = check_points_within_in_second_camera_frustrum(image_dim, world_points_camera2, RT1_w2c, K1, max_depth)

        iou = (np.sum(camera1_pts_visible_camera2) + np.sum(camera2_pts_visible_camera1)) / (
                camera1_pts_visible_camera2.shape[0] + camera2_pts_visible_camera1.shape[0])

        return iou





def rescale_intrinsics(intrinsics, old_resolution, new_resolution):
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


def compute_frustum_overlap_with_mesh(mesh, RT1_w2c, K1, RT2_w2c, K2, image_size=(224, 224)):
    """
    Computes overlap between two cameras given a mesh, intrinsics, and extrinsics.
    The overlap is based on shared visible mesh points.
    """

    img_pts1 =  project_world_point2_image_pixels(mesh.vertices, RT1_w2c, K1)
    img_pts2 = project_world_point2_image_pixels(mesh.vertices, RT2_w2c, K2)

    valid_mask1 =  (np.round(img_pts1[0,:]).astype(np.int64)>=0) & (np.round(img_pts1[0, :]).astype(np.int64) < image_size[1]) & (np.round(img_pts1[1,:].astype(np.int64))>=0)& (np.round(img_pts1[1, :]).astype(np.int64) <  image_size[0]) & (img_pts1[2,:]>0.0)
    valid_mask2 =  (np.round(img_pts2[0,:]).astype(np.int64)>=0) & (np.round(img_pts2[0, :]).astype(np.int64) < image_size[1]) & (np.round(img_pts2[1,:].astype(np.int64))>=0)& (np.round(img_pts2[1, :]).astype(np.int64) <  image_size[0]) & (img_pts2[2,:]>0.0)


    # Compute overlap as intersection over union
    visible_both = valid_mask1 & valid_mask2
    union = valid_mask1 | valid_mask2

    overlap = visible_both.sum() / union.sum() if union.sum() > 0 else 0.0
    return overlap


def get_matched_keypoints(
    images,
    name_to_id,
    points3D,
    img1_name,
    img2_name,
    original_res=(480, 256),
    target_res=(224, 224)
):
    """
    Given COLMAP images and 3D points, and two image names,
    returns the shared 2D keypoints between the two views, rescaled to target resolution.

    Parameters:
        images (dict): Mapping from image_id to Image objects (from read_images_binary)
        points3D (dict): Mapping from 3D point_id to Point3D objects (from read_points3D_binary)
        img1_name (str): Name of the first image (e.g., "frame_0001.jpg")
        img2_name (str): Name of the second image (e.g., "frame_0002.jpg")
        original_res (tuple): (width, height) at which COLMAP keypoints were computed
        target_res (tuple): (width, height) for rescaling keypoints

    Returns:
        dict: {
            'img1_kp': np.ndarray of shape (N, 2),
            'img2_kp': np.ndarray of shape (N, 2)
        }
    """

    # Compute scale factors
    scale_x = target_res[0] / original_res[0]
    scale_y = target_res[1] / original_res[1]

    # Map image names to image IDs
    name_to_id = {img.name: img_id for img_id, img in images.items()}
    img1_id = name_to_id[img1_name]
    img2_id = name_to_id[img2_name]

    img1 = images[img1_id]
    img2 = images[img2_id]

    points1 = []
    points2 = []


    for point in points3D.values():
        if img1_id in point.image_ids and img2_id in point.image_ids:
            idx1 = list(point.image_ids).index(img1_id)
            idx2 = list(point.image_ids).index(img2_id)

            kp1 = img1.xys[point.point2D_idxs[idx1]]
            kp2 = img2.xys[point.point2D_idxs[idx2]]

            # Rescale to target resolution
            kp1_rescaled =[kp1[0] * scale_x, kp1[1] * scale_y]
            kp2_rescaled = [kp2[0] * scale_x, kp2[1] * scale_y]

            points1.append(kp1_rescaled)
            points2.append(kp2_rescaled)

    return {
        "img1_kp": points1,
        "img2_kp": points2
    }


def compute_overlaps(DATA_ROOT, start_idx, end_idx):
    """
    Compute the overlap between camera frustums for a subset of files given by `file_range`.
    """
    SAVE_ROOT = "/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/sparse_overlaps/"
    SPARSE_ROOT  = "/scratch/projects/fouheylab/shared_datasets/epic-fields/Sparse/"
    IMAGES_ROOT  = ""
    camera_ids = [
        "P24_05", "P03_04", "P01_14", "P30_107", "P05_08", "P12_101", "P28_103", 
        "P10_04", "P30_05", "P06_101", "P04_05", "P06_103", "P35_109", "P37_103", 
        "P04_11", "P04_21", "P04_109", "P02_07", "P28_14", "P22_01", "P15_02", 
        "P04_26", "P01_09", "P02_109", "P02_101", "P24_08", "P23_05", "P28_110", 
        "P20_03", "P11_105", "P08_09", "P22_07", "P03_113", "P04_02", "P25_107", 
        "P02_130", "P08_16", "P30_101", "P18_07", "P01_103", "P01_05", "P03_03", 
        "P11_102", "P06_107", "P03_24", "P37_101", "P06_12", "P02_107", "P03_17", 
        "P01_104", "P11_16", "P06_13", "P02_122", "P06_11", "P28_109", "P03_101", 
        "P02_124", "P03_05", "P04_114", "P28_06", "P03_123", "P02_121", "P27_101", 
        "P03_13", "P06_07", "P26_110", "P03_112", "P30_112", "P04_33", "P02_135", 
        "P02_03", "P04_101", "P12_02", "P02_102", "P05_01", "P01_03", "P22_117", 
        "P17_01", "P06_09", "P03_11", "P28_101", "P06_110", "P04_04", "P28_13", 
        "P30_111", "P18_06", "P28_113", "P03_23", "P11_101", "P32_01", "P04_121", 
        "P04_110", "P12_03", "P04_25", "P08_21", "P02_128", "P04_03", "P14_05", 
        "P23_02", "P28_112", "P06_01", "P07_08", "P11_103", "P02_132", "P06_14", 
        "P02_01", "P18_03", "P06_102", "P01_01", "P35_105"
    ]

    npz_files = [f for f in os.listdir(DATA_ROOT) if f.endswith(".npz") and f.split('.')[0] in camera_ids]
    npz_files.sort()
    files_to_process = npz_files[start_idx:end_idx]

    with open('/scratch/projects/fouheylab/dma9300/OSNOM/diverse_frames.json', 'r') as f:
            diverse_frame_dict = json.load(f)

    for file in files_to_process:
        overlaps = {}

        full_path = os.path.join(DATA_ROOT, file)
        kitchen = file.split(".")[0]
        mesh_path = os.path.join(MESH_ROOT_PATH, kitchen, 'fused-minpix15-meshed-delaunay-qreg5.ply')
        points3D = read_write_model.read_points3D_binary(os.path.join(SPARSE_ROOT, kitchen, "sparse", "0", "points3D.bin"))
        images_colmap  = read_write_model.read_images_binary(os.path.join(SPARSE_ROOT, kitchen, "sparse", "0", "images.bin"))
        img_name_to_id = {img.name: img_id for img_id, img in images_colmap.items()}
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_scale(rescale_scores[kitchen])

        try:
            data = np.load(full_path, allow_pickle=True)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        camera_keys = sorted(list(data.keys()))

        sampled_keys =diverse_frame_dict[kitchen]

        for image_name1 in sampled_keys:
            data1 = data[image_name1]
            RT1_w2c = np.linalg.inv(data1.item()["RT"])
            intrinsics1 = rescale_intrinsics(data1.item()["intrinsics"], (480, 256), (224, 224))
            overlaps[image_name1] = {}

            for image_name2 in sampled_keys:
                if image_name1 == image_name2:
                    continue
                data2 = data[image_name2]
                RT2_w2c = np.linalg.inv(data2.item()["RT"])
                intrinsics2 = rescale_intrinsics(data2.item()["intrinsics"], (480, 256), (224, 224))

                overlap = compute_frustum_overlap_with_mesh(
                    mesh,
                    RT1_w2c=RT1_w2c,
                    K1=intrinsics1,
                    RT2_w2c=RT2_w2c,
                    K2=intrinsics2,
                    image_size=(224, 224),
                )
                matching_points  = get_matched_keypoints(images_colmap, img_name_to_id, points3D, image_name1, image_name2)

                if len(matching_points["img1_kp"]) >= 5 and len(matching_points["img2_kp"]) >= 5:
                    overlaps[image_name1][image_name2] = {
                        "overlap": overlap,
                        "kp1": matching_points["img1_kp"],
                        "kp2": matching_points["img2_kp"]
                    }
    
            if not overlaps[image_name1]:
                del overlaps[image_name1]
        
        # Save overlaps to JSON file
        overlaps_json_path = f'{SAVE_ROOT}/{file.split(".")[0]}_overlaps.json'
        with open(overlaps_json_path, "w") as f:
            json.dump(overlaps, f, indent=4)

        print(f"Saved overlaps to {overlaps_json_path}")



def main():
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Compute camera frustum overlaps.")
    parser.add_argument('--start_idx', type=int, required=True, help="Start index for the file range.")
    parser.add_argument('--end_idx', type=int, required=True, help="End index for the file range.")
    parser.add_argument('--root', type=str, default="/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/sparse_cameras/",
                        help="Root directory containing the .npz files.")
    
    args = parser.parse_args()

    # Call the compute_overlaps function with the parsed arguments
    compute_overlaps(args.root, args.start_idx, args.end_idx)


if __name__ == "__main__":
    random.seed(42)  # Set seed for reproducibility
    main()
    








