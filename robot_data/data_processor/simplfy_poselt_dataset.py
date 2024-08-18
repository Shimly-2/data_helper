from .BaseDataProcessor import BaseDataProcessor
import cv2
import time
import os
import json
from tqdm import tqdm
from tqdm import trange
import math
import multiprocessing
import os.path as osp
import numpy as np
import copy
from robot_data.utils.registry_factory import DATA_PROCESSER_REGISTRY
from robot_data.utils.robot_timestamp import RobotTimestampsIncoder
from robot_data.utils.utils import get_dirpath_from_key
import csv

def dict2list(data):
    result = []
    if isinstance(data, (list, np.ndarray, tuple)):
        for item in data:
            result.extend(dict2list(item))
    elif isinstance(data, dict):
        for value in data.values():
            result.extend(dict2list(value))
    else:
        result.append(data)
    return result

@DATA_PROCESSER_REGISTRY.register("SimplfyPoseltDataset")
class SimplfyPoseltDataset(BaseDataProcessor):
    """生产meta数据，将所有视频，图片路径变为绝对路径"""
    def __init__(
        self,
        workspace,
        dataset_root,
        new_dataset_root,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.timestamp_maker = RobotTimestampsIncoder()
        self.new_dataset_root = new_dataset_root

    def gen_per_meta(self, meta_info, json_path):
        with open(os.path.join(self.dataset_root, json_path),"r") as f:
            meta_info = json.load(f)
        print(f"PID[{os.getpid()}: Load json file - {json_path}")

        case_name = json_path.split("/")[-2]
        object_name = case_name.split('_')[0]
        # root_path = get_dirpath_from_key(json_path, "raw_meta")
        # frame_infos = meta_info["frame_info"]
        # window_set = []
        for idx in tqdm(meta_info["frame_info"], colour='green', desc=f'PID[{os.getpid()}]'):
            for cam_view, img_path_ in meta_info["frame_info"][idx]["cam_views"].items():
                img_path = img_path_["rgb_img_path"]
                old_img_path = os.path.join(self.dataset_root, img_path)
                new_img_path = os.path.join(self.new_dataset_root, img_path)
                new_img_dir = os.path.dirname(new_img_path)
                os.makedirs(new_img_dir, exist_ok=True)
                old_img = cv2.imread(old_img_path)
                cv2.imwrite(new_img_path, old_img)
            for tactile_view, img_path_ in meta_info["frame_info"][idx]["tactile_views"].items():
                img_path = img_path_["rgb_img_path"]
                old_img_path = os.path.join(self.dataset_root, img_path)
                new_img_path = os.path.join(self.new_dataset_root, img_path)
                new_img_dir = os.path.dirname(new_img_path)
                os.makedirs(new_img_dir, exist_ok=True)
                old_img = cv2.imread(old_img_path)
                cv2.imwrite(new_img_path, old_img)
        with open(os.path.join(self.new_dataset_root, json_path), "w") as f:
            json.dump(meta_info, f) #, indent=4)
        return meta_info
        
    def process(self, meta, task_infos):
        results = []
        json_paths = []
        json_dir_list = [name for name in os.listdir(self.dataset_root) if os.path.isdir(os.path.join(self.dataset_root, name))]
        if self.pool > 1:
            args_list = []
            meta_info = []
            for json_dir in json_dir_list:  #依次读取视频文件
                json_path = os.path.join(json_dir, "result.json")
                args_list.append((meta_info, json_path))
            results = self.multiprocess_run(self.gen_per_meta, args_list)
        else:
            meta_info = []
            for idx, json_dir in enumerate(json_dir_list):  #依次读取视频文件
                filename = osp.split(json_dir)[-1]
                json_path = os.path.join(json_dir, "result.json")
                # save_new_filename = osp.join(self.save_root, filename)
                self.logger.info(
                    f"Start process {idx+1}/{len(json_dir_list)}")
                results.append(
                    self.gen_per_meta(meta_info, json_path))
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            if meta_infos == []:
                pass
            else:
                meta.append(meta_infos)
        return meta, task_infos
