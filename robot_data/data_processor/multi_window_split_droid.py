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

@DATA_PROCESSER_REGISTRY.register("MulitWindowSplitDROID")
class MulitWindowSplitDROID(BaseDataProcessor):
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

    def merge_regen(self, e_idx, episode, root_path):
        meta_info = []
        window_set = []
        save_root = os.path.join(root_path, "window_set")
        os.makedirs(save_root, exist_ok=True)
        for idx, step in enumerate(tqdm(episode["steps"], colour='green', desc=f'PID[{os.getpid()}]')):
            per_window = dict()
            per_window["ds_idx"] = e_idx
            per_window["window_start"] = int(idx)
            per_window["window_end"] = int(idx) + self.window_sec
            if per_window["window_end"] > len(episode["steps"]):
                break
            if not self.check_instructions_continues(per_window["window_start"], per_window["window_end"], episode["steps"]):
                continue
            per_window["instruction"] = step["language_instruction"].numpy().decode('utf-8')
            per_window["label"] = "move"
            per_window["joint"] = dict2list(step["observation"]["joint_position"].numpy())
            per_window["joint"].append(list(step["observation"]["gripper_position"].numpy())[0])
            window_set.append(per_window)
        with open(os.path.join(save_root, f"{self.split_set}_e_{e_idx}_window_set.json"), "w") as f:
            json.dump(window_set, f)
        return meta_info
        
    def process(self, meta, task_infos):
        results = []
        ds = tfds.load(self.dataset_key, data_dir=self.dataset_root, split=self.split_set)
        root_path = os.path.join(self.dataset_root, self.dataset_key)
        if self.pool > 1:
            args_list = []
            meta_info = []
            for e_idx, episode in enumerate(ds):
                args_list.append((e_idx, episode, root_path))
            results = self.multiprocess_run(self.merge_regen, args_list)
        else:
            meta_info = []
            for e_idx, episode in enumerate(ds):
                self.logger.info(
                    f"Start process {e_idx+1}/{len(ds)}")
                results.append(
                    self.merge_regen(e_idx, episode, root_path))
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            meta.append(meta_infos)
        return meta, task_infos
