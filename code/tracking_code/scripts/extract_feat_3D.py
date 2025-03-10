import os
import sys
import subprocess

# Constants
BASE_PATH = '/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/'
OUTPUT_DIR = '.'
EPIC_KITCHENS_PATH = './data/EPIC-KITCHENS/'

# Videos List
VIDEOS = ['P24_05',
          'P03_04',
          'P01_14',
          'P30_107',
          'P05_08',
          'P12_101',
          'P28_103',
          'P10_04',
          'P30_05',
          'P06_101',
          'P04_05',
          'P06_103',
          'P35_109',
          'P37_103',
          'P04_11',
          'P04_21',
          'P04_109',
          'P02_07',
          'P28_14',
          'P22_01',
          'P15_02',
          'P04_26',
          'P01_09',
          'P02_109',
          'P02_101',
          'P24_08',
          'P23_05',
          'P28_110',
          'P20_03',
          'P11_105',
          'P08_09',
          'P22_07',
          'P03_113',
          'P04_02',
          'P25_107',
          'P02_130',
          'P08_16',
          'P30_101',
          'P18_07',
          'P01_103',
          'P01_05',
          'P03_03',
          'P11_102',
          'P06_107',
          'P03_24',
          'P37_101',
          'P06_12',
          'P02_107',
          'P03_17',
          'P01_104',
          'P11_16',
          'P06_13',
          'P02_122',
          'P06_11',
          'P28_109',
          'P03_101',
          'P02_124',
          'P03_05',
          'P04_114',
          'P28_06',
          'P03_123',
          'P02_121',
          'P27_101',
          'P03_13',
          'P06_07',
          'P26_110',
          'P03_112',
          'P30_112',
          'P04_33',
          'P02_135',
          'P02_03',
          'P04_101',
          'P12_02',
          'P02_102',
          'P05_01',
          'P01_03',
          'P22_117',
          'P17_01',
          'P06_09',
          'P03_11',
          'P28_101',
          'P06_110',
          'P04_04',
          'P28_13',
          'P30_111',
          'P18_06',
          'P28_113',
          'P03_23',
          'P11_101',
          'P32_01',
          'P04_121',
          'P04_110',
          'P12_03',
          'P04_25',
          'P08_21',
          'P02_128',
          'P04_03',
          'P14_05',
          'P23_02',
          'P28_112',
          'P06_01',
          'P07_08',
          'P11_103',
          'P02_132',
          'P06_14',
          'P02_01',
          'P18_03',
          'P06_102',
          'P01_01',
          'P35_105']

# Command Template
BASE_COMMAND = [
    "python",
    os.path.join(BASE_PATH, "code/tracking_code/extract_feat/save_feat_batch_3D.py"),
    "--output_dir", OUTPUT_DIR,
]

def get_full_command(video):
    """Construct the full command for the given video."""
    first_number = video.split('_')[0]
    return BASE_COMMAND + [
        "--data_path", os.path.join(BASE_PATH, "data/aggregated", video),
        "--mesh_path", os.path.join(BASE_PATH, "data/aggregated", video),
        "--frames_path", os.path.join(BASE_PATH, "preprocessed_data/",  f"{video}.tar"),
        "--kitchen", video,
    ]


# Main Execution
for video in VIDEOS:
    print(f"Processing video: {video}")
    full_command = get_full_command(video)
    print("Executing command:", full_command)
    subprocess.run(full_command)
