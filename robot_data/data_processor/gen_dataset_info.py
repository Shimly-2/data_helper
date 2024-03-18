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

@DATA_PROCESSER_REGISTRY.register("GenDatasetINFO")
class GenDatasetINFO(BaseDataProcessor):
    """生产dataset数据"""
    def __init__(
        self,
        workspace,
        dataset_root,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.timestamp_maker = RobotTimestampsIncoder()

    def gen_per_meta(self, object_name, json_path):
        dataset_info = dict()
        dataset_info["episode_length"] = []
        json_path = sorted(json_path)
        for per_json_path in json_path:
            with open(per_json_path, "r") as f:
                meta_info = json.load(f)
            print(f"PID[{os.getpid()}: Load json file - {per_json_path}")
            dataset_info["episode_length"].append(meta_info["case_info"]["steps"])
        root_path = get_dirpath_from_key(json_path[0], object_name)
        root_path = root_path.replace("raw_meta", "train_meta")
        root_path = os.path.join(root_path, object_name)
        os.makedirs(root_path, exist_ok=True)
        with open(os.path.join(root_path, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=4)
        return meta_info
        
    def process(self, meta, task_infos):
        results = []
        json_paths = dict()
        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in filenames:
                if filename.endswith("result.json"):
                    object_name = dirpath.split("/")[-2]
                    if object_name not in json_paths:
                        json_paths[object_name] = []
                        json_paths[object_name].append(os.path.join(dirpath, filename))
                    else:
                        json_paths[object_name].append(os.path.join(dirpath, filename))
        if self.pool > 1:
            args_list = []
            for object_name, json_path in json_paths.items():  #依次读取视频文件
                args_list.append((object_name, json_path))
            results = self.multiprocess_run(self.gen_per_meta, args_list)
        else:
            idx = 0
            for object_name, json_path in json_paths.items():  #依次读取视频文件
                filename = osp.split(json_path)[-1]
                # save_new_filename = osp.join(self.save_root, filename)
                self.logger.info(
                    f"Start process {idx+1}/{len(json_paths)}")
                results.append(
                    self.gen_per_meta(object_name, json_path))
                idx = idx + 1
        return meta, task_infos
