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
from collections import defaultdict

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

def dictkey2list(data):
    result = []
    if isinstance(data, (list, np.ndarray, tuple)):
        for item in data:
            result.extend(dict2list(item))
    elif isinstance(data, dict):
        for value in data.keys():
            result.extend(dict2list(value))
    else:
        result.append(data)
    return result

@DATA_PROCESSER_REGISTRY.register("FixError")
class FixError(BaseDataProcessor):
    """get all the label and turn to one-hot"""
    def __init__(
        self,
        workspace,
        dataset_root,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.timestamp_maker = RobotTimestampsIncoder()

    def gen_per_meta(self, meta_info, json_path):
        with open(os.path.join(self.dataset_root, json_path), "r") as f:
            data = json.load(f)
        print(f"PID[{os.getpid()}: Load json file - {json_path}")
        for idx in tqdm(data["frame_info"], colour='green', desc=f'PID[{os.getpid()}]'):
            for cam_view, img_path_ in data["frame_info"][idx]["cam_views"].items():
                if data["frame_info"][idx]["cam_views"][cam_view]["rgb_img_path"][0] == "/":
                    data["frame_info"][idx]["cam_views"][cam_view]["rgb_img_path"] = data["frame_info"][idx]["cam_views"][cam_view]["rgb_img_path"][1:]
                if data["frame_info"][idx]["cam_views"][cam_view]["depth_img_path"][0] == "/":
                    data["frame_info"][idx]["cam_views"][cam_view]["depth_img_path"] = data["frame_info"][idx]["cam_views"][cam_view]["depth_img_path"][1:]
            for tactile_view, img_path_ in data["frame_info"][idx]["tactile_views"].items():
                if data["frame_info"][idx]["tactile_views"][tactile_view]["rgb_img_path"][0] == "/":
                    data["frame_info"][idx]["tactile_views"][tactile_view]["rgb_img_path"] = data["frame_info"][idx]["tactile_views"][tactile_view]["rgb_img_path"][1:]
            if len(data["frame_info"][idx]["commended"]["ee_command_orientation"]) == 2:
                data["frame_info"][idx]["commended"]["ee_command_orientation"] = [-np.pi] + data["frame_info"][idx]["commended"]["ee_command_orientation"]
        
        # object_name = json_path.split("/")[-3]
        # for cam_view, option in data["cogagent_label"].items():
        #     data["cogagent_label"][cam_view]["object"] = object_name
        # print(data["frame_info"]["0"]["cam_views"])
        # print(data["frame_info"]["0"]["tactile_views"])
        # print(data["frame_info"]["0"]["commended"]["ee_command_orientation"])
        # print(data["cogagent_label"])

        with open(os.path.join(self.dataset_root, json_path), "w") as f:
            json.dump(data, f) #, indent=4)
        return meta_info
        
    def process(self, meta, task_infos):
        results = []
        json_paths = []
        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in filenames:
                if filename.endswith("result.json"):
                    json_paths.append(os.path.join(dirpath, filename))
        if self.pool > 1:
            args_list = []
            meta_info = []
            for json_path in json_paths:  #依次读取视频文件
                args_list.append((meta, json_path))
            results = self.multiprocess_run(self.gen_per_meta, args_list)
        else:
            meta_info = []
            for idx, json_path in enumerate(json_paths):  #依次读取视频文件
                self.logger.info(
                    f"Start process {idx+1}/{len(json_paths)}")
                results.append(
                    self.gen_per_meta(meta, json_path))
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            meta.append(meta_infos)

        return meta, task_infos
