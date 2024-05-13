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
import tensorflow_datasets as tfds

@DATA_PROCESSER_REGISTRY.register("ShuffleWindowSetDROID")
class ShuffleWindowSetDROID(BaseDataProcessor):
    """根据window set生成训练集和验证集, 无多线程"""
    def __init__(
        self,
        workspace,
        dataset_root,
        window_sec,
        split_set,
        dataset_key,
        random_seed,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.split_set = split_set
        self.random_seed = random_seed
        self.window_sec = window_sec
        self.dataset_key = dataset_key
        self.timestamp_maker = RobotTimestampsIncoder()      

    def check_instructions_continues(self, start, end, steps):
        start_l = ""
        steps = steps.skip(start).take(end)
        for idx, step in enumerate(steps):
            if idx == 0:
                start_l = step["language_instruction"]
            else:
                if step["language_instruction"] != start_l:
                    return False
        return True

    def merge_regen(self, json_paths, split_set, root_path):
        meta_info = []
        window_set = []
        for json_path in tqdm(json_paths, colour='green', desc=f'PID[{os.getpid()}]'):
            with open(json_path,"r") as f:
                meta_info = json.load(f)
            window_set.extend(meta_info)
        random.seed(self.random_seed)
        random.shuffle(window_set)
        with open(os.path.join(root_path, f"{split_set}_sec_{self.window_sec}_seed_{self.random_seed}.json"), "w") as f:
            json.dump(window_set, f) #, indent=4)
        self.logger.info(f"{split_set} num: {len(window_set)}")         
        return meta_info
        
    def process(self, meta, task_infos):
        results = []
        json_paths = []
        root_path = os.path.join(self.dataset_root, self.dataset_key)
        for _set in self.split_set:
            for dirpath, dirnames, filenames in os.walk(self.dataset_root):
                for filename in filenames:
                    if filename.endswith("window_set.json") and _set in filename:
                        json_paths.append(os.path.join(dirpath, filename))
            self.merge_regen(json_paths, _set, root_path)
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            meta.append(meta_infos)
        return meta, task_infos
