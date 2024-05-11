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

@DATA_PROCESSER_REGISTRY.register("MultiWindowSplit")
class MultiWindowSplit(BaseDataProcessor):
    """根据meta分割视频为不同窗口"""
    def __init__(
        self,
        workspace,
        dataset_root,
        window_sec,
        target_cam_view,
        target_tactile_view,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.window_sec = window_sec
        self.target_cam_view = target_cam_view
        self.target_tactile_view = target_tactile_view
        self.timestamp_maker = RobotTimestampsIncoder()
    
    def check_instructions_continues(self, start, end, frame_infos):
        for idx in range(start, end):
            if frame_infos[str(idx)]["instruction"] != frame_infos[str(start)]["instruction"]:
                return False
        return True


    def split_per_meta(self, meta_info, json_path):
        with open(json_path,"r") as f:
            meta_info = json.load(f)
        print(f"PID[{os.getpid()}: Load json file - {json_path}")
        case_name = meta_info["case_info"]["case_name"]
        task_name = meta_info["case_info"]["task_name"]
        case_name_evl = json_path.split("/")[-5]
        task_name_evl = json_path.split("/")[-6]
        assert case_name == case_name_evl, "Not the same case: [meta]--<{}>, [config]--<{}>".format(case_name, case_name_evl)
        assert task_name == task_name_evl, "Not the same task: [meta]--<{}>, [config]--<{}>".format(task_name, task_name_evl)
        root_path = get_dirpath_from_key(json_path, "raw_meta")
        frame_infos = meta_info["frame_info"]
        window_set = []
        for idx in tqdm(frame_infos, colour='green', desc=f'PID[{os.getpid()}]'):
            per_window = dict()
            per_window["window_start"] = int(idx)
            per_window["window_end"] = int(idx) + self.window_sec
            if per_window["window_end"] > len(frame_infos):
                break
            if not self.check_instructions_continues(per_window["window_start"], per_window["window_end"], frame_infos):
                continue
            frame_info = frame_infos[idx]
            per_window["instruction"] = frame_info["instruction"]
            cam_view_info = frame_info["cam_views"][self.target_cam_view]
            cam_view_info["rgb_video_path"] = os.path.join(root_path, cam_view_info["rgb_video_path"]).replace(".mp4", "#s224.mp4")
            per_window["video_path"] = cam_view_info["rgb_video_path"]
            tactile_view_info = frame_info["tactile_views"][self.target_tactile_view]
            tactile_view_info["rgb_video_path"] = os.path.join(root_path, tactile_view_info["rgb_video_path"]).replace(".mp4", "#s224.mp4")
            per_window["tactile_path"] = tactile_view_info["rgb_video_path"]
            per_window["label"] = "move"
            per_window["joint"] = dict2list(frame_info["jointstates"])[:7]
            per_window["joint"].append(frame_info["commended"]["gripper_closedness_commanded"])
            window_set.append(per_window)
        with open(os.path.join(os.path.dirname(json_path), "window_set.json"), "w") as f:
            json.dump(window_set, f) #, indent=4)
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
            results = self.multiprocess_run(self.split_per_meta, args_list)
        else:
            meta_info = []
            for idx, json_path in enumerate(json_paths):  #依次读取视频文件
                filename = osp.split(json_path)[-1]
                # save_new_filename = osp.join(self.save_root, filename)
                self.logger.info(
                    f"Start process {idx+1}/{len(json_paths)}")
                results.append(
                    self.split_per_meta(meta_info, json_path))
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            meta.append(meta_infos)
        return meta, task_infos
