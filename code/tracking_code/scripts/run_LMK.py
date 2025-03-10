import sys
sys.path.append('./code/tracking_code')

import os
import subprocess
import wandb
from tracking.tracker import PHALP

# Base Paths
BASE_PATH = '.'
RESULTS_PATH = os.path.join(BASE_PATH, '')
TUNE_OUTPUT_PATH = os.path.join(RESULTS_PATH, 'results.pkl')


# Parameters

# Weight assigned to appearance features in the model.
BETA_0 = 2  
# Weight assigned to location features in the model.
BETA_1 = 13  
# Number of past appearance features to consider for aggregation.
PAST_LOOKBACK = 100  
# Threshold used in the Hungarian algorithm for assignment problems.
# Determines when associations are made or rejected based on costs.
HUNGARIAN_TH = 10  
# Type of distance metric to use:
# - 'A0': Considers only appearance features.
# - 'L0': Considers only location features.
# - 'AL': Combines appearance and location features.
DISTANCE_TYPE = 'AL'


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

# Helper Functions
def get_sweep_config(video):
    return {
        "method": "grid",
        "parameters": {
            "use_velocity": {"value": False},
            "activation": {"values": ["sigmoid"]},
            "T": {"value": 1},
            "beta_0": {"values": [BETA_0]},
            "beta_1": {"values": [BETA_1]},
            "beta_2": {"values": [1.0]},
            "beta_3": {"values": [1.0]},
            "output_dir": {"values": [RESULTS_PATH]},
            "kitchen": {"values": [video]},
            "use_unproj": {"values": [False]},
            "visualize": {"values": [False]},
            "save_res": {"values": [True]},
            "random": {"values": [False]},
            "n_init": {"values": [5]},
            "predict": {"values": ["A"]},
            "distance_type": {"values": [DISTANCE_TYPE]},
            "alpha": {"values": [0.1]},
            "hungarian_th": {"values": [HUNGARIAN_TH]},
            "track_history": {"values": [7]},
            "max_age_track": {"values": [10000000]},
            "encode_type": {"values": ["4c"]},
            "past_lookback": {"values": [PAST_LOOKBACK]},
            "detection_type": {"values": ["mask"]},
            "shot": {"values": [0]},
            "aggregation": {"values": ["mean"]},
            "model": {"values": ["dino"]},
            "use_pred": {"value": False},
            "experiment_name": {"value": "demo"},
            "track_dataset": {"value": "demo"},
            "device": {"value": "cuda"},
            "train": {"value": False},
            "debug": {"value": False},
            "use_gt": {"value": False},
            "batch_id": {"value": -1},
            "verbose": {"value": False},
            "detect_shots": {"value": False},
            "base_path": {"value": BASE_PATH},
            "video_seq": {"value": None},
            "dir_name": {"value": 'results_AL/'}
        }
    }

def should_skip_results(video):
    results_file = TUNE_OUTPUT_PATH.format(
        past_lookback=PAST_LOOKBACK, video=video, beta_0=BETA_0, beta_1=BETA_1
    )
    return os.path.exists(results_file)


# Main Execution
for video in VIDEOS:
    if not should_skip_results(video):
        wandb.login()
        sweep_config = get_sweep_config(video)
        sweep_id = wandb.sweep(sweep_config, project="tuning")

        phalp = PHALP()
        wandb.agent(sweep_id, phalp.track)
