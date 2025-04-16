import numpy as np
from sklearn.cluster import KMeans
import os
import json

def load_camera_poses_from_sparse_cameras(tar_folder, sparse_camera_folder):
    """
    Extract video IDs from .tar files and load camera poses from corresponding .npz files.
    
    Returns:
        dict: { video_id: { frame_id: camera_to_world_pose (4x4) } }
    """
    all_camera_poses = {}

    for file in os.listdir(tar_folder):
        if not file.endswith(".tar"):
            continue

        video_id = os.path.splitext(file)[0]
        camera_file = os.path.join(sparse_camera_folder, f"{video_id}.npz")

        if not os.path.exists(camera_file):
            print(f"Skipping {video_id}: No camera file found at {camera_file}")
            continue

        try:
            data = np.load(camera_file, allow_pickle=True)
            # breakpoint()
        except Exception as e:
            print(f"Error reading {camera_file}: {e}")
            continue

        cam_poses = {}
        for frame_id, frame_data in data.items():
            # breakpoint()
            RT_c2w = np.array(frame_data.item()["RT"])  # Already camera-to-world
            cam_poses[frame_id] = RT_c2w

        all_camera_poses[video_id] = cam_poses

    return all_camera_poses


def pose_to_position(pose):
    """Extract 3D camera position from 4x4 pose matrix."""
    return pose[:3, 3]

def select_diverse_frames_by_kmeans(camera_poses, num_clusters=20):
    """Select diverse frames using KMeans clustering."""
    frame_ids = list(camera_poses.keys())
    positions = np.array([pose_to_position(camera_poses[fid]) for fid in frame_ids])

    if len(frame_ids) <= num_clusters:
        return frame_ids

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(positions)
    centers = kmeans.cluster_centers_

    selected_frames = []
    for center in centers:
        dists = np.linalg.norm(positions - center, axis=1)
        closest_idx = np.argmin(dists)
        selected_frames.append(frame_ids[closest_idx])

    return selected_frames

def build_diverse_frame_dict(all_camera_poses, num_clusters=20):
    """
    Build dictionary: video_id -> list of diverse frame IDs.
    
    Args:
        all_camera_poses (dict): {video_id: {frame_id: pose_matrix}}
        num_clusters (int): number of diverse frames to select per video.

    Returns:
        dict: {video_id: [diverse_frame_id1, ...]}
    """
    video_to_diverse_frames = {}

    for video_id, cam_poses in all_camera_poses.items():
        diverse_frames = select_diverse_frames_by_kmeans(cam_poses, num_clusters=num_clusters)
        video_to_diverse_frames[video_id] = diverse_frames

    return video_to_diverse_frames

def save_diverse_frame_dict(diverse_frame_dict, filename="diverse_frames.json"):
    """
    Save the diverse frame dictionary to a JSON file.
    
    Args:
        diverse_frame_dict (dict): The dictionary of diverse frames.
        filename (str): The name of the file to save the dictionary.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(diverse_frame_dict, f, indent=4)
        print(f"Diverse frame dictionary saved to {filename}")
    except Exception as e:
        print(f"Error saving dictionary: {e}")

if __name__ == "__main__":
    # Set folder paths (change as needed)
    tar_folder = "/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/preprocessed_data"
    sparse_cameras_folder = "/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/sparse_cameras"

    # Load camera poses from .npz files corresponding to video IDs
    all_camera_poses = load_camera_poses_from_sparse_cameras(tar_folder, sparse_cameras_folder)

    # Build the dictionary of diverse frames for each video
    diverse_frame_dict = build_diverse_frame_dict(all_camera_poses, num_clusters=40)

    # Save the diverse frame dictionary to a JSON file in the current directory
    save_diverse_frame_dict(diverse_frame_dict, filename="diverse_frames.json")