#! /bin/bash

# Usage: ./code/mesh_generation_code/download_images.sh <video_id>

# Exit on error or unset variable
set -eu

VIDEO_ID=$1
echo $VIDEO_ID

# Paths
TEMP_TAR_DIR="data/EpicKitchens-100"
IMAGES_DIR="data/aggregated/images"

# Download data if token does not exist
images_downloaded_token="$IMAGES_DIR/$VIDEO_ID.done"
if [ ! -f "$images_downloaded_token" ]; then
    echo "Downloading images for $VIDEO_ID"
    
    # video number os part after `_`
    vid_num=$(echo "$VIDEO_ID" | cut -d'_' -f2)

    # Does the tarball already exist?
    person_id=$(echo "$VIDEO_ID" | cut -d'_' -f1)    # person id (pid) is the part before `_`
    tarball_dir="$TEMP_TAR_DIR/$person_id/rgb_frames/"
    tarball_path="$tarball_dir/$VIDEO_ID.tar"
    tarball_done_token="$VIDEO_ID.done"

    # Store current directory, mkdir, and cd to tarball directory
    curr_dir=$(pwd)
    mkdir -p $tarball_dir
    cd $tarball_dir

    # If tarball does not exist, download it
    if [ ! -f "$tarball_done_token" ]; then
        echo "Downloading tarball for $VIDEO_ID"
        # If video number is less than 100, use one link, else use another
        if [ $vid_num -lt 100 ]; then
            wget -c https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/train/$person_id/$VIDEO_ID.tar || wget -c https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/$person_id/$VIDEO_ID.tar
        else
            wget -c https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/$person_id/rgb_frames/$VIDEO_ID.tar
        fi
        touch $tarball_done_token
    else
        echo "Tarball for $VIDEO_ID already exists"
    fi

    # cd back to original directory
    cd $curr_dir

    # Extract the frames and delete tarball
    echo "Extracting frames for $VIDEO_ID"
    mkdir -p $IMAGES_DIR/$VIDEO_ID
    tar -xf $tarball_path -C $IMAGES_DIR/$VIDEO_ID
    rm $tarball_path

    touch $images_downloaded_token
else
    echo "Images for $VIDEO_ID already exists"
fi
