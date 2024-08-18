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

@DATA_PROCESSER_REGISTRY.register("AnalyzsPoselt")
class AnalyzsPoselt(BaseDataProcessor):
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
        label_count = dict()
        label_count["grasp_label"] = dict()
        label_count["action_stage"] = dict()
        with open(os.path.join(self.dataset_root, json_path), "r") as f:
            data = json.load(f)
        print(f"PID[{os.getpid()}: Load json file - {json_path}")
        for idx in tqdm(data["frame_info"], colour='green', desc=f'PID[{os.getpid()}]'):
            label_count["grasp_label"][data["frame_info"][idx]["grasp_label"]] = 1
            label_count["action_stage"][data["frame_info"][idx]["action_stage"]] = 1
            # data["frame_info"][idx]["action_stage"] = "move"
        meta_info = label_count
        # with open(json_path, "w") as f:
        #     json.dump(meta_info, f) #, indent=4)
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
            meta.append(meta_infos)
        grasp_label_idx = dict()
        for item in meta:
            grasp_label = item.get('grasp_label', {})
            for key in grasp_label.keys():
                grasp_label_idx[key] = 1 # tmp.get(key, 0) + 1
        action_stage_idx = dict()
        for item in meta:
            action_stage = item.get('action_stage', {})
            for key in action_stage.keys():
                action_stage_idx[key] = 1 # tmp.get(key, 0) + 1
        meta = [
            {
                "grasp_label_idx": grasp_label_idx,
                "action_stage_idx": action_stage_idx,
            }
        ]
        # import pdb;pdb.set_trace()
        return meta, task_infos
