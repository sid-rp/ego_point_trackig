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
import tarfile

boxes_scales = {'P24_05': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_04': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P01_14': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P30_107': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P05_08': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P12_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P28_103': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P10_04': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P30_05': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_05': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_103': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P35_109': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P37_103': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_11': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_21': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_109': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_07': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P28_14': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P22_01': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P15_02': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_26': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P01_09': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_109': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P24_08': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P23_05': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P28_110': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P20_03': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P11_105': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P08_09': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P22_07': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_113': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_02': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P25_107': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_130': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P08_16': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P30_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P18_07': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P01_103': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P01_05': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_03': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P11_102': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_107': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_24': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P37_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_12': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_107': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_17': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P01_104': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P11_16': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_13': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_122': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_11': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P28_109': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_124': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_05': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_114': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P28_06': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_123': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_121': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P27_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_13': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_07': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P26_110': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_112': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P30_112': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_33': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_135': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_03': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P12_02': {'x_scale': 0.3563, 'y_scale': 0.3556},
 'P02_102': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P05_01': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P01_03': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P22_117': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P17_01': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_09': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_11': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P28_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_110': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_04': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P28_13': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P30_111': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P18_06': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P28_113': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P03_23': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P11_101': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P32_01': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_121': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_110': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P12_03': {'x_scale': 0.3563, 'y_scale': 0.3556},
 'P04_25': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P08_21': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_128': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P04_03': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P14_05': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P23_02': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P28_112': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_01': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P07_08': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P11_103': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_132': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_14': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P02_01': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P18_03': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P06_102': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P01_01': {'x_scale': 0.2375, 'y_scale': 0.237},
 'P35_105': {'x_scale': 0.2375, 'y_scale': 0.237}}


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
    
    def read_image_from_tar(self, tar_file_path, file_name):
        """
        Reads an image file from a .tar archive and converts it to OpenCV format (NumPy array).
        
        Parameters:
            tar_file_path (str): Path to the .tar archive.
            file_name (str): Name of the image file inside the tar archive to read.
        
        Returns:
            numpy.ndarray: The image as a NumPy array in OpenCV format.
        """

        try:
            # Open the tar file in read mode
            with tarfile.open(tar_file_path, "r") as tar:
                # Try to extract the specific image file
                file = tar.extractfile(f"./{file_name}")
                
                if file:
                    # Read the image content as bytes
                    image_bytes = file.read()
                    
                    # Convert the bytes to a NumPy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    
                    # Decode the NumPy array into an OpenCV image
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Check if the image was successfully decoded
                    if img is None:
                        print(f"Failed to decode the image {file_name}.")
                        return None
                    return img
                else:
                    print(f"File {file_name} not found inside the tar archive.")
                    return None
        except KeyError:
            print(f"The file {file_name} does not exist inside the tar archive.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def rescale_all_bounding_boxes(self, bboxes, x_scale, y_scale):
            """
            Rescales all bounding boxes in the list according to the given x and y scale factors.

            Parameters:
            bboxes (list of tuples/lists): List of bounding boxes, where each bounding box is represented 
                                            as (x_min, y_min, x_max, y_max).
            x_scale (float): Scale factor for the x-coordinates.
            y_scale (float): Scale factor for the y-coordinates.

            Returns:
            list of tuples: List of rescaled bounding boxes.
            """
            rescaled_bboxes = []
            
            # Loop through each bounding box and rescale it
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                x_min_rescaled = x_min * x_scale
                y_min_rescaled = y_min * y_scale
                x_max_rescaled = x_max * x_scale
                y_max_rescaled = y_max * y_scale
                rescaled_bboxes.append((x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled))

            return rescaled_bboxes


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

        self.data_path = f"{self.base_path}/aggregated/{self.cfg.kitchen}/"
        self.frames_path = f"{self.base_path}/preprocessed_data/{self.cfg.kitchen}.tar"
        # breakpoint()
        with open(os.path.join(self.data_path, 'poses.json'), 'r') as f:
            self.poses = json.load(f)
        self.masks, _, self.camera_poses, self.frames, _ = read_data_1(self.data_path, '', self.cfg.kitchen, True)

        with open(f"{self.base_path}/osnom_3d_features/saved_feat_3D/{self.cfg.kitchen}/3D_feat_{self.cfg.kitchen}.pkl", 'rb') as file:
            self.all_loca = pickle.load(file)
        with open(f"{self.base_path}/osnom_2d_features/saved_feat_2D/{self.cfg.kitchen}/2D_feat_{self.cfg.kitchen}.pkl", 'rb') as file:
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
                bbs  = self.rescale_all_bounding_boxes(bbs,boxes_scales[self.kitchen]["x_scale"], boxes_scales[self.kitchen]["y_scale"] )

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
                # breakpoint()
                cv_image = self.read_image_from_tar(self.frames_path,f"{frame_name}.jpg")
                # cv_image = cv2.imread(os.path.join(self.frames_path, f"{frame_name}.jpg"))
                cv_image = cv2.resize(cv_image, (854, 480))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                for bbox, tr_id in zip(final_visuals_dic[frame_name]['tracked_bbox'], final_visuals_dic[frame_name]['tracked_ids']):
                    cv_image = visualize_mask(cv_image, None, bbox, color=np.array(self.RGB_tuples[tr_id]), text=f"track id: {tr_id}")
                img = Image.fromarray(cv_image)
                img.save(os.path.join(self.path_to_save, f"{frame_name}.jpg"))

            else:
                print('No detections')