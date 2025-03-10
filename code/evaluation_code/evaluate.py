import sys
import math
from tqdm import tqdm
import pandas as pd
import argparse
import pickle
import os

def load_results(results_path):
    with open(results_path, 'rb') as f:
        return pickle.load(f)

def extract_frame_number(frame_name):
    return int(frame_name.split('_')[1])


def propagate_annotations_across_frames(save_dict, set_ids=None, present_ids=None):
    """
    Propagates annotations across frames, ensuring that annotations are updated in all previous frames where they are present.

    Args:
        save_dict (dict): Dictionary to save the propagated annotations.
        set_ids (list, optional): List of set IDs to propagate. Defaults to None.
        present_ids (dict, optional): Dictionary indicating the presence of IDs in each frame. Defaults to None.
    """
    first_frame_for_annotation = {}
    done_list = []
    last = sorted(save_dict.keys())[-1]
    keys = sorted(save_dict.keys())

    for frame_name in tqdm(sorted(save_dict.keys())):
        annotations = set_ids if set_ids else save_dict[last].keys()
        for annotation in annotations:
            if annotation not in first_frame_for_annotation:
                first_frame_for_annotation[annotation] = frame_name
            if annotation in save_dict[frame_name].keys() and annotation not in done_list:
                current_frame = frame_name
                first_frame = first_frame_for_annotation[annotation]
                first_frame_index = keys.index(first_frame)
                current_frame_index = keys.index(current_frame)
                for i in range(first_frame_index, current_frame_index):
                    frame = keys[i]
                    if present_ids is None or annotation in present_ids[frame]:
                        save_dict[frame][annotation] = save_dict[current_frame][annotation]
                done_list.append(annotation)
    return save_dict

def get_object_from_results(results, all_frames):
    """
    Processes results and all_frames to generate dictionaries for bounding boxes, names, and locations.

    Args:
        results (dict): Dictionary containing tracking results with frame names as keys.
        all_frames (list): List of all frame names.

    Returns:
        tuple: Tuple containing:
            - bbs_dict (dict): Dictionary of bounding box lists for each frame.
            - bbs_dict_name (dict): Dictionary of annotation names for each frame.
            - save_dict_gt_id (dict): Dictionary of ground truth IDs for each annotation in each frame.
            - save_dict_gt_loca (dict): Dictionary of ground truth locations for each annotation in each frame.
            - save_dict_id_loca (dict): Dictionary of locations for each tracked ID in each frame.
    """
    bbs_dict = {}
    bbs_dict_name = {}
    save_dict_gt_id = {}
    save_dict_gt_loca = {}
    save_dict_id_loca = {}
    present_ids = {}
    last_frame = None
    set_ids = []

    save_dicts = [save_dict_id_loca, save_dict_gt_loca, save_dict_gt_id]

    for frame_name in sorted(all_frames):
        obj_list = []
        name_list = []
        for save_dict in save_dicts:
            save_dict[frame_name] = save_dict[last_frame].copy() if last_frame else {}

        if frame_name in results:
            for e, annotation in enumerate(results[frame_name]['tracked_gt']):
                name_list.append(annotation)
                tracked_id = results[frame_name]['tracked_ids'][e]
                location = results[frame_name]['loca'][e]
                if tracked_id not in set_ids:
                    set_ids.append(tracked_id)
                save_dict_gt_id[frame_name][annotation] = tracked_id
                save_dict_gt_loca[frame_name][annotation] = location
                save_dict_id_loca[frame_name][tracked_id] = location

            # Note: check if the ID is associated to an object in that frame
            for tracked_id in list(save_dict_id_loca[frame_name]):
                if tracked_id not in save_dict_gt_id[frame_name].values():
                    del save_dict_id_loca[frame_name][tracked_id]

        last_frame = frame_name
        bbs_dict[frame_name] = obj_list
        bbs_dict_name[frame_name] = name_list

    save_dict_gt_id = propagate_annotations_across_frames(save_dict_gt_id)
    for frame_name in save_dict_gt_id.keys():
      present_ids[frame_name] = save_dict_gt_id[frame_name].values()
    save_dict_gt_loca = propagate_annotations_across_frames(save_dict_gt_loca)
    save_dict_id_loca = propagate_annotations_across_frames(save_dict_id_loca, set_ids, present_ids)


    return bbs_dict, bbs_dict_name, save_dict_gt_id, save_dict_gt_loca, save_dict_id_loca



def euclidean_distance(loc1, loc2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(loc1, loc2)))


def add_frames(frame_name, N):
    # Extract the numeric part of the frame name
    frame_number = int(frame_name.split('_')[-1])

    # Add N frames to the frame number
    new_frame_number = frame_number + N

    # Create the new frame name with leading zeros
    new_frame_name = f'frame_{new_frame_number:010d}'

    return new_frame_name


def sub_frames(frame_name, N):
    # Extract the numeric part of the frame name
    frame_number = int(frame_name.split('_')[-1])

    # Add N frames to the frame number
    new_frame_number = frame_number - N

    # Create the new frame name with leading zeros
    new_frame_name = f'frame_{new_frame_number:010d}'

    return new_frame_name


def calculate_percentage_correct_with_start(mid_frames, n, R, gt_id, gt_loca, id_loca):
    """
    Calculate the percentage of correctly tracked objects within a specified distance threshold R.

    Args:
        mid_frames (dict): Dictionary with keys as frame indices and values as lists of elements (objects).
        n (int): Number of frames to add or subtract to get the target frame.
        R (float): Distance threshold for considering a tracked object as correct.
        gt_id (dict): Dictionary of ground truth IDs for each annotation in each frame.
        gt_loca (dict): Dictionary of ground truth locations for each annotation in each frame.
        id_loca (dict): Dictionary of locations for each tracked ID in each frame.

    Returns:
        float: The percentage of correctly tracked objects within the distance threshold R. Returns -1 if there are no objects to evaluate.
    """
    correct_count = 0
    total_count = 0
    for elements, mid_frame_key in tqdm(mid_frames.items()):
        for element in elements:
            start_idx = mid_frame_key
            target_idx_add = add_frames(start_idx, n)
            target_idx_sub = sub_frames(start_idx, n)

            for target_idx in [target_idx_add, target_idx_sub]:
                if target_idx in gt_id.keys():
                    current_id = gt_id[start_idx][element]
                    current_gt = element

                    if current_gt in gt_loca[target_idx].keys():
                        loca_gt = gt_loca[target_idx][current_gt]


                        if current_id in id_loca[target_idx]:
                          loca_id = id_loca[target_idx][current_id]
                          dist = euclidean_distance(loca_gt, loca_id)
                          if dist <= R:
                              correct_count += 1
                        total_count += 1

    if total_count > 0:
        percentage_correct = (correct_count / total_count) * 100
    else:
        percentage_correct = -1

    return percentage_correct

def compute_mid_frames(data):
    sets = []
    dict_set = {}
    for f in data.keys():
        if len(data[f]) > 2:
            sets.append(data[f])
            if tuple(sorted(data[f])) not in dict_set:
                dict_set[tuple(sorted(data[f]))] = []
            dict_set[tuple(sorted(data[f]))].append(f)

    mid_frame = {}
    if len(dict_set.keys()) == 0:
        for f in data.keys():
            if len(data[f]) >= 2:
                sets.append(data[f])
                if tuple(sorted(data[f])) not in dict_set:
                    dict_set[tuple(sorted(data[f]))] = []
                dict_set[tuple(sorted(data[f]))].append(f)
    for s in dict_set.keys():
        mid_frame[s] = dict_set[s][len(dict_set[s]) // 2]
    return mid_frame


def print_results(results_grid, R, N):
    print("\t", end="\t")
    for r in R:
        print(f"r={r}", end="\t")
    print()

    for n in N:
        print(f"n={n}:", end="\t")
        for r in R:
            percentage = results_grid[n][r]
            print(f"{percentage:.2f}%", end="\t")
        print()


def save_results_to_csv(filename, results_grid, R, N):
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header = ["n"] + [f"r={r}" for r in R]
        writer.writerow(header)
        for n in N:
            row_data = [f"n={n}"] + [results_grid[n][r] for r in R]
            writer.writerow(row_data)

def load_results(results_path):
    with open(results_path, 'rb') as f:
        return pickle.load(f)

def list_all_frames(directory_path):
    return [f.split('.')[0] for f in os.listdir(directory_path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True, help='Path to results file')
    parser.add_argument('--frames_path', type=str, required=True, help='Path to directory')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--kitchen', type=str, required=True, help='Path to data')
    parser.add_argument('--video', type=str, required=True, help='Path to data')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to data')
    parser.add_argument('--name_prefix', type=str, required=True, help='Path to data')

    print('parse args...')
    args = parser.parse_args()
    df = pd.read_csv('./data/EPIC_100_video_info.csv')
    fps = int(df[df['video_id'] == args.kitchen + '_' + args.video]['fps'])
    fps = 60 if fps == 59 else fps


    R = [1]
    N = [fps * 5 * i for i in range(144)]
    results_grid = {}

    print('load results...')
    results = load_results(args.results_path)
    print('results loaded...')

    all_frames = list_all_frames(args.frames_path)


    # Prepare results for the evaluation
    gts, names, gt_id, gt_loca, id_loca = get_object_from_results(results, all_frames)

    # Compute initial frames for evaluation
    mid_frame = compute_mid_frames(names)

    for n in N:
        results_grid[n] = {}

        for r in R:
            percentage = calculate_percentage_correct_with_start(
                mid_frame,
                n, r,
                gt_id,
                gt_loca, id_loca
            )
            results_grid[n][r] = percentage
            
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_grid, f)
                    
if __name__ == "__main__":
    main()
