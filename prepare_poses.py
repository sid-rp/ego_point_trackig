import os
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from read_write_model import read_images_binary  # from COLMAP's scripts

# def qvec_to_rotmat(qvec):
#     """Convert COLMAP quaternion to rotation matrix using scipy"""
#     return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

# def process_colmap_dir(colmap_root):
#     colmap_root = Path(colmap_root)
    
#     for subdir in colmap_root.iterdir():
#         sparse_dir = subdir / "sparse" / "0"  # Keep it Path objects all the way
#         images_bin = sparse_dir / "images.bin"
        
#         if not images_bin.exists():
#             print(f"Skipping {subdir.name}: no images.bin found.")
#             continue
        
#         try:
#             images = read_images_binary(str(images_bin))
#         except Exception as e:
#             print(f"Failed to read {images_bin}: {e}")
#             continue

#         poses = {}
#         for image in images.values():
#             R_mat = qvec_to_rotmat(image.qvec)
#             t_vec = image.tvec.reshape(3, 1)
#             RT = np.eye(4)
#             RT[:3, :3] = R_mat
#             RT[:3, 3] = t_vec[:, 0]

#             poses[image.name] = RT

#         # Save all RTs as a .npz file
#         root_path  = "/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/sparse_camera_poses"
#         out_path =f"{root_path}/{subdir.name}_poses.npz"
#         np.savez(out_path, **poses)
#         print(f"Saved RT poses to {out_path}")


import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from read_write_model import read_images_binary, read_cameras_binary  # from COLMAP

def qvec_to_rotmat(qvec):
    return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

def process_colmap_dir(colmap_root):
    colmap_root = Path(colmap_root)

    for subdir in colmap_root.iterdir():
        sparse_dir = subdir / "sparse" / "0"
        images_bin = sparse_dir / "images.bin"
        cameras_bin = sparse_dir / "cameras.bin"
      
        if not images_bin.exists() or not cameras_bin.exists():
            print(f"Skipping {subdir.name}: missing COLMAP files.")
            continue

        try:
            images = read_images_binary(str(images_bin))
            cameras = read_cameras_binary(str(cameras_bin))
        except Exception as e:
            print(f"Error reading COLMAP files in {subdir.name}: {e}")
            continue

        poses = {}
        for image in images.values():
            
            cam = cameras[image.camera_id]

            # RT matrix
            R_mat = qvec_to_rotmat(image.qvec)
            t_vec = image.tvec.reshape(3, 1)
            RT = np.eye(4)
            RT[:3, :3] = R_mat
            RT[:3, 3] = t_vec[:, 0]

            # Intrinsics: create a 3x3 intrinsic matrix based on OpenCV format
            f_x, f_y = cam.params[0], cam.params[1]  # focal lengths
            c_x, c_y = cam.params[2], cam.params[3]  # principal point
            skew = 0  # skew is typically 0 for most cameras

            intrinsic_matrix = np.array([
                [f_x, skew, c_x],
                [0, f_y, c_y],
                [0, 0, 1]
            ])

            poses[image.name] = {
                "RT": RT,
                "intrinsics": intrinsic_matrix
            }
        
    
        #Save all RTs as a .npz file
        root_path  = "/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/sparse_cameras"
        out_path =f"{root_path}/{subdir.name}.npz"
        np.savez(out_path, **poses)
        print(f"Saved RT poses to {out_path}")
process_colmap_dir("/scratch/projects/fouheylab/shared_datasets/epic-fields/Sparse/")
