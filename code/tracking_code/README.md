## Blender Visualization
See [blender/README.md](blender/README.md) for details on Blender visualization scripts.

## 3D Mesh reconstruction
COLMAP scripts for parallel reconstruction of 3D meshes, given sparse resonctruction and images:
```bash
sbatch sbatch_reconstruct_rense.sh

# Please set the following paths in the script:
# Used by COLMAP:
# IMAGE_PATH="data/images/$VID"
# CAMERAS_PATH="colmap_models/sparse/$VID/sparse/0"
# DENSE3D_PATH="colmap_models/dense3D/$VID/"
# COLMAP_COMMAND="singularity exec --nv --bind /home --bind /old_home_that_will_be_deleted_at_some_point/ /old_home_that_will_be_deleted_at_some_point/shubham/colmap.sif colmap"

# Others
# input_videos2.txt                             # Set of videos to run on
# images_downloaded_token                       # token marking whther images for a video have been downloaded and extracted
# tarball_dir                                   # where we download image tars
# tar -xf $tarball_path -C data/images/$vid     # images extraced here
```
