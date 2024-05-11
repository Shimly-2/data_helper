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
from robot_data.utils.utils import get_dirpath_from_key, dict2list
import random

@DATA_PROCESSER_REGISTRY.register("ShuffleWindowSet")
class ShuffleWindowSet(BaseDataProcessor):
    """根据window set生成训练集和验证集, 无多线程"""
    def __init__(
        self,
        workspace,
        dataset_root,
        window_sec,
        split_set,
        random_seed,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.split_set = split_set
        self.random_seed = random_seed
        self.window_sec = window_sec
        self.timestamp_maker = RobotTimestampsIncoder()
    
    def divideTrainValTest(self, window_set, root_path, split_set):
        train_list = window_set[0:int(split_set[0] * 0.1 * len(window_set))]
        val_list = window_set[int(split_set[0] * 0.1 * len(window_set)):int((split_set[0] + split_set[1]) * 0.1 * len(window_set))]
        test_list = window_set[int((split_set[0] + split_set[1]) * 0.1 * len(window_set)):]
        with open(os.path.join(root_path, f"train_sec_{self.window_sec}_seed_{self.random_seed}.json"), "w") as f:
            json.dump(train_list, f) #, indent=4)
        with open(os.path.join(root_path, f"val_sec_{self.window_sec}_seed_{self.random_seed}.json"), "w") as f:
            json.dump(val_list, f) #, indent=4)
        with open(os.path.join(root_path, f"test_sec_{self.window_sec}_seed_{self.random_seed}.json"), "w") as f:
            json.dump(test_list, f) #, indent=4)
        self.logger.info(f"Train num: {len(train_list)}, Val num: {len(val_list)}, Test num: {len(test_list)}")

    def merge_regen(self, json_paths):
        window_set = []
        root_path = get_dirpath_from_key(json_paths[0], "raw_meta")
        for json_path in tqdm(json_paths, colour='green', desc=f'PID[{os.getpid()}]'):
            with open(json_path,"r") as f:
                meta_info = json.load(f)
            # print(f"PID[{os.getpid()}: Load json file - {json_path}")
            window_set.extend(meta_info)
        random.seed(self.random_seed)
        random.shuffle(window_set)    
        self.divideTrainValTest(window_set, root_path, self.split_set)
        return meta_info
        
    def process(self, meta, task_infos):
        results = []
        json_paths = []
        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in filenames:
                if filename.endswith("window_set.json"):
                    json_paths.append(os.path.join(dirpath, filename))
        self.merge_regen(json_paths)
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            meta.append(meta_infos)
        return meta, task_infos
