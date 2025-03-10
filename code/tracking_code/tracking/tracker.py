import torch.nn as nn
import numpy as np
import torch
import pickle
import json
import os
import cv2
from tqdm import tqdm
from PIL import Image
import wandb
from external.deep_sort_ import nn_matching
from external.deep_sort_.detection import Detection
from external.deep_sort_.tracker import Tracker
from util import get_colors, read_data_1, get_object_bbs_seg, visualize_mask


items_dict = {
    "P22_01": ["plate", "jar", "drawer", "knife", "fork", "cup", "sponge", "banana", "peach", "place mat"],
    "P24_05": ["cupboard door", "cupboard", "knife"],
    "P03_04": ["bag", "squash/squash drink/juice concentrate", "food", "onion", "drawer", "knife", "kitchen towel",
               "lid", "can", "spice", "spoon"],
    "P01_14": ["bin/garbage can/recycling bin", "plate", "lid", "glass"],
    "P30_107": ["bowl", "saucepan", "sponge"],
    "P05_08": ["saucepan"],
    "P12_101": ["meat box", "cheese", "cup", "spoon", "mug", "cream cheese container"],
    "P28_103": ["sponge", "cup", "towel", "drawer", "bin"],
    "P10_04": [],
    "P30_05": ["plate", "box", "bag", "drawer", "cupboard", "bottle", "bowl", "surface", "carrot"],
    "P06_101": ["pepper", "drawer", "onion", "sweet potato", "pot", "cupboard", "lid", "oil"],
    "P04_05": ["wok", "bag", "drawer", "spoon", "lid", "tray"],
    "P06_103": ["lid", "pot", "drawer", "glove", "cupboard", "bowl", "cup", "glass"],
    "P35_109": ["bowl", "container", "cupboard", "plate", "drawer", "mug", "cup"],
    "P37_103": [],
    "P04_11": [],
    "P04_21": [],
    "P04_109": ["carrot", "cucumber"],
    "P02_07": [],
    "P28_14": [],
    "P15_02": ["cupboard"],
    "P04_26": [],
    "P01_09": ["bottle", "potato", "cupboard", "drawer", "bowl", "flour", "cloth", "plate", "lid", "paper"],
    "P02_109": ["chopping board", "meat"],
    "P02_101": ["lid"],
    "P24_08": ["cupboard"],
    "P23_05": ["plate"],
    "P28_110": [],
    "P20_03": [],
    "P11_105": ["oven tray", "pizza", "pizza box"],
    "P08_09": ["kitchen towel", "coffee", "lid", "avocado"],
    "P22_07": ["glass", "cloth", "cupboard", "drawer", "sponge", "rag", "lid", "plate"],
    "P03_113": ["saucepan", "plate", "rice"],
    "P04_02": ["pan", "bowl", "egg", "drawer", "tuna patty", "cupboard"],
    "P25_107": ["pan", "plate", "aubergine", "cupboard"],
    "P02_130": ["spoon", "tin"],
    "P08_16": ["bowl", "spoon", "egg", "cupboard"],
    "P30_101": ["cupboard", "mug", "plate", "sponge", "drawer"],
    "P18_07": ["cupboard", "plate"],
    "P01_103": ["cupboard", "plate", "knife", "spoon"],
    "P01_05": ["plate", "cupboard", "bowl", "knife", "chopping board", "spoon"],
    "P03_03": [],
    "P11_102": ["leek"],
    "P06_107": [],
    "P03_24": ["saucepan", "plate"],
    "P37_101": ["stool", "cupboard", "bowl"],
    "P06_12": [],
    "P02_107": [],
    "P03_17": ["saucepan", "plate", "lid"],
    "P01_104": [],
    "P11_16": [],
    "P06_13": ["pot", "glove", "lid"],
    "P02_122": ["glove", "plate", "cupboard"],
    "P06_11": ["lid", "pot"],
    "P28_109": ["drawer"],
    "P03_101": ["plate", "saucepan"],
    "P02_124": ["washing powder box", "drawer", "box"],
    "P03_05": ["saucepan", "lid"],
    "P04_114": [],
    "P28_06": [],
    "P03_123": ["oven mitt", "clip top jar"],
    "P02_121": ["spoon"],
    "P27_101": ["cupboard", "pan", "drawer", "lid", "spatula", "cloth", "glass", "plate"],
    "P03_13": [],
    "P06_07": ["plate", "pizza", "dough", "pizza base"],
    "P03_112": ["cupboard"],
    "P30_112": [],
    "P04_33": ["plate"],
    "P02_135": [],
    "P02_03": ["pan", "lid", "cupboard", "towel", "plate"],
    "P04_101": ["pan", "cupboard", "plate", "onion"],
    "P12_02": ["knife"],
    "P02_102": ["box", "oil bottle"],
    "P05_01": ["cup"],
    "P01_03": [],
    "P22_117": ["drawer", "cupboard", "bag", "glass", "lid"],
    "P17_01": ["bowl", "lid"],
    "P06_09": [],
    "P03_11": [],
    "P28_101": ["knife", "sponge"],
    "P06_110": [],
    "P04_04": ["drawer", "packet", "frying pan", "cupboard"],
    "P28_13": ["egg"],
    "P30_111": ["cupboard", "mug", "pan", "sponge", "wooden spoon"],
    "P18_06": ["bowl", "cupboard", "plate", "cup"],
    "P28_113": ["cupboard", "drawer", "knife"],
    "P03_23": ["drawer"],
    "P11_101": [],
    "P32_01": ["pan"],
    "P04_121": ["cupboard", "whey", "tea towel"],
    "P04_110": ["cupboard", "plate", "tea towel"],
    "P12_03": ["spoon", "fork"],
    "P04_25": [],
    "P08_21": ["drawer", "glass", "bowl", "kitchen towel", "mug", "knife", "cupboard", "lid", "plate", "container",
               "spoon", "counter", "milk bottle"],
    "P02_128": ["box", "milk bottle", "lid"],
    "P04_03": ["pan", "tuna burger", "bin/garbage can/recycling bin", "lunch box", "tuna patty", "kitchen roll",
               "knife", "bowl"],
    "P14_05": [],
    "P23_02": ["counter", "plate"],
    "P28_112": ["bag", "counter"],
    "P06_01": ["cupboard", "cereal"],
    "P07_08": ["tortilla", "plate"],
    "P11_103": ["drawer", "pan"],
    "P02_132": ["tupperware", "tupperware lid"],
    "P06_14": [],
    "P02_01": ["cupboard", "plate"],
    "P18_03": ["plate", "cupboard"],
    "P06_102": ["cupboard", "knife"],
    "P35_105": ["cloth", "saucepan", "container", "cupboard", "plate", "salt box"]
}


class PHALP(nn.Module):

    def __init__(self):
        super(PHALP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Device:', self.device)
        self.to(self.device)
        self.eval()

    def setup_deepsort(self):
        print("Setting up DeepSort...")
        metric = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.hungarian_th, self.cfg.past_lookback)
        self.tracker = Tracker(self.cfg, metric, max_age=self.cfg.max_age_track, n_init=self.cfg.n_init,
                               phalp_tracker=self, dims=[4096, 4096, 99])

    def track(self, config=None, debug=False):
        wb = wandb.init(config=config)
        wb.name = f"output_{wandb.config.distance_type}_a{wandb.config.alpha}_h{wandb.config.hungarian_th}_pl{wandb.config.past_lookback}_agg{wandb.config.aggregation}_{wandb.config.model}_{wandb.config.kitchen}_beta0{wandb.config.beta_0}_beta1{wandb.config.beta_1}"
        self.cfg = wb.config
        self.save_res = self.cfg.save_res
        self.path_to_save = os.path.join(self.cfg.output_dir, self.cfg.dir_name, "tune_output")
        os.makedirs(self.path_to_save, exist_ok=True)
        self.RGB_tuples = get_colors()
        self.kitchen = self.cfg.kitchen
        self.base_path = self.cfg.base_path
        first_number = self.kitchen.split('_')[0]

        self.data_path = f"{self.base_path}/data/aggregated/{self.cfg.kitchen}/"
        self.frames_path = f"{self.base_path}/EPIC-KITCHENS/{first_number}/rgb_frames/{self.cfg.kitchen}/"
        with open(os.path.join(self.data_path, 'poses.json'), 'r') as f:
            self.poses = json.load(f)
        self.masks, _, self.camera_poses, self.frames, _ = read_data_1(self.data_path, '', self.cfg.kitchen, True)

        with open(f"{self.base_path}/saved_feat_3D/{self.cfg.kitchen}/3D_feat_{self.cfg.kitchen}.pkl", 'rb') as file:
            self.all_loca = pickle.load(file)
        with open(f"{self.base_path}/saved_feat_2D/{self.cfg.kitchen}/2D_feat_{self.cfg.kitchen}.pkl", 'rb') as file:
            self.all_feat = pickle.load(file)

        self.bbs_dict = get_object_bbs_seg(self.masks['video_annotations'])
        visual_store_ = ['tracked_ids', 'tracked_bbox', 'tracked_gt', 'tid', 'bbox', 'tracked_time', 'features', 'loca', 'radius', 'size', 'img_path', 'img_name', 'conf']
        final_visuals_dic = {}
        tracked_frames = []

        self.setup_deepsort()

        for t_, frame_name in tqdm(enumerate(sorted(self.frames)), total=len(self.frames)):
            _, bbs, objs = self.bbs_dict.get(f"{self.cfg.kitchen}_{frame_name}", (None, [], []))
            if t_ == len(self.frames) - 1 and self.save_res:
                with open(os.path.join(self.path_to_save, 'results.pkl'), 'wb') as f:
                    pickle.dump(final_visuals_dic, f)

            detections = []
            removed_indices = []

            if bbs:
                try:
                    appe_features = self.all_feat[frame_name]
                    features_3d = self.all_loca[frame_name]
                except KeyError:
                    print('Frame features not found!')
                    with open(os.path.join(self.path_to_save, 'results.pkl'), 'wb') as f:
                        pickle.dump(final_visuals_dic, f)
                    continue

                if self.cfg.kitchen in items_dict:
                    duplicates = items_dict[self.cfg.kitchen]
                    gt_copy = objs.copy()
                    for item in duplicates:
                        if item in gt_copy:
                            index = gt_copy.index(item)
                            objs.remove(item)
                            removed_indices.append(index)
                    bbs = [b for i, b in enumerate(bbs) if i not in removed_indices]

                feat_3D = np.delete(features_3d[0], removed_indices, axis=0)
                radius = np.delete(features_3d[1], removed_indices, axis=0)
                appe = np.delete(appe_features, removed_indices, axis=0)

                for i in range(len(objs)):
                    detection_data = {
                        "bbox": np.array([bbs[i][0], bbs[i][1], (bbs[i][2] - bbs[i][0]), (bbs[i][3] - bbs[i][1])]),
                        "conf": 1.0,
                        "appe": appe[i],
                        "loca": feat_3D[i],
                        "radius": radius[i],
                        "size": [480, 854],
                        "img_path": frame_name[0] + "/" + frame_name[1],
                        "img_name": frame_name[1],
                        "ground_truth": objs[i],
                        "time": t_,
                    }
                    detections.append(Detection(detection_data))

            self.tracker.predict()
            _, statistics = self.tracker.update(detections, t_, frame_name)

            final_visuals_dic.setdefault(frame_name, {'time': t_, 'frame': frame_name})
            for key_ in visual_store_:
                final_visuals_dic[frame_name].setdefault(key_, [])
            for track in self.tracker.tracks:
                if frame_name not in tracked_frames:
                    tracked_frames.append(frame_name)
                track_id = track.track_id
                track_data_hist = track.track_data['history'][-1]
                if track.time_since_update == 0:
                    final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                    final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                    final_visuals_dic[frame_name]['tracked_gt'].append(track_data_hist['ground_truth'])
                    final_visuals_dic[frame_name]['loca'].append(track_data_hist['loca'])
                    final_visuals_dic[frame_name]['radius'].append(track_data_hist['radius'])

            if self.cfg.visualize and final_visuals_dic[frame_name]['tracked_ids']:
                cv_image = cv2.imread(os.path.join(self.frames_path, f"{frame_name}.jpg"))
                cv_image = cv2.resize(cv_image, (854, 480))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                for bbox, tr_id in zip(final_visuals_dic[frame_name]['tracked_bbox'], final_visuals_dic[frame_name]['tracked_ids']):
                    cv_image = visualize_mask(cv_image, None, bbox, color=np.array(self.RGB_tuples[tr_id]), text=f"track id: {tr_id}")
                img = Image.fromarray(cv_image)
                img.save(os.path.join(self.path_to_save, f"{frame_name}.jpg"))

            else:
                print('No detections')