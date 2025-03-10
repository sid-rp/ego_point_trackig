#! /bin/bash

# Usage: ./code/mesh_generation_code/reconstruct_mesh.sh <video_id>

# Exit on error or unset variable
set -eu

VIDEO_ID=$1
echo $VIDEO_ID

# Paths
IMAGE_PATH="data/images/$VIDEO_ID"
CAMERAS_PATH="data/colmap_models/sparse/$VIDEO_ID/sparse/0"
DENSE3D_PATH="data/colmap_models/dense3D/$VIDEO_ID/"

mkdir -p $DENSE3D_PATH

# =========
# Hyperparameters
minpix=15
# =========

# Skip whole shebang if final mesh already exists
FINAL_MESH_PATH="$DENSE3D_PATH/fused-minpix$minpix-meshed-delaunay-qreg5.ply"
if [ -f "$FINAL_MESH_PATH" ]; then
    echo "Final mesh already exists for $VIDEO_ID at $FINAL_MESH_PATH"
    exit 0
fi

# COLMAP_COMMAND="singularity exec --fakeroot --nv --bind $PWD colmap.sif colmap" # Built from https://gist.github.com/shubham-goel/31b63f6b7499a7d61090a9e32a9f1a26
# COLMAP_COMMAND="singularity exec --nv --bind $PWD docker://colmap/colmap:latest colmap"
COLMAP_COMMAND="colmap"

echo "colmap image_undistorter"
$COLMAP_COMMAND image_undistorter \
    --image_path $IMAGE_PATH \
    --input_path $CAMERAS_PATH \
    --output_path $DENSE3D_PATH \
    --output_type COLMAP \
    --copy_policy soft-link \
    --max_image_size 2000

echo "colmap patch_match_stereo"
$COLMAP_COMMAND patch_match_stereo \
    --workspace_path $DENSE3D_PATH \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.gpu_index 0

echo "colmap stereo_fusion (min pix$minpix)"
$COLMAP_COMMAND stereo_fusion \
    --workspace_path $DENSE3D_PATH \
    --workspace_format COLMAP \
    --input_type geometric \
    --StereoFusion.min_num_pixels $minpix \
    --output_path $DENSE3D_PATH/fused-minpix$minpix.ply

echo "colmap poisson_mesher"
$COLMAP_COMMAND poisson_mesher \
    --input_path $DENSE3D_PATH/fused-minpix$minpix.ply \
    --output_path $DENSE3D_PATH/fused-minpix$minpix-meshed-poisson-d10-t5.ply \
    --PoissonMeshing.depth 10 \
    --PoissonMeshing.num_threads 5 \
    --PoissonMeshing.trim 5

echo "copying fused-minpix$minpix.ply to fused.ply"
cp $DENSE3D_PATH/fused-minpix$minpix.ply $DENSE3D_PATH/fused.ply
cp $DENSE3D_PATH/fused-minpix$minpix.ply.vis $DENSE3D_PATH/fused.ply.vis
echo "colmap delaunay_mesher"
$COLMAP_COMMAND delaunay_mesher \
    --input_path $DENSE3D_PATH \
    --output_path $DENSE3D_PATH/fused-minpix$minpix-meshed-delaunay-qreg5.ply \
    --output_path $FINAL_MESH_PATH \
    --DelaunayMeshing.quality_regularization 5
