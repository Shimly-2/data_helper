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
            if idx == "cogagent_label":
                continue
            label_count["grasp_label"][data["frame_info"][idx]["grasp_label"]] = 1
            label_count["action_stage"][data["frame_info"][idx]["action_stage"]] = 1
            # data["frame_info"][idx]["action_stage"] = "move"
        meta_info = label_count
        cogagent_label = data["cogagent_label"]
        meta_info.update({
            "cogagent_label": cogagent_label
        })
        # with open(json_path, "w") as f:
        #     json.dump(meta_info, f) #, indent=4)
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
                args_list.append((meta_info, json_path))
            results = self.multiprocess_run(self.gen_per_meta, args_list)
        else:
            meta_info = []
            for idx, json_path in enumerate(json_paths):  #依次读取视频文件
                self.logger.info(
                    f"Start process {idx+1}/{len(json_paths)}")
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
        cogagent_label_idx = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # 统计每个 key 的 value 出现次数
        for item in meta:
            cogagent_label = item.get('cogagent_label', {})
            for cam_view, subdict in cogagent_label.items():
                for option, value in subdict.items():
                    if isinstance(value, list):
                        value = value[0]
                    if "image" in value or "picture" in value or " is" in value:
                        continue
                    cogagent_label_idx[cam_view][option][value] = 1

        # 将 defaultdict 转换为普通字典以便于打印
        cogagent_label_idx = {k: dict(v) for k, v in cogagent_label_idx.items()}
        cogagent_label_idx = {k: {subk: dict(subv) for subk, subv in v.items()} for k, v in cogagent_label_idx.items()}
        meta = [
            {
                "grasp_label_idx": grasp_label_idx,
                "action_stage_idx": action_stage_idx,
                "cogagent_label_idx": cogagent_label_idx,
            }
        ]
        print(meta)
        # import pdb;pdb.set_trace()
        return meta, task_infos
