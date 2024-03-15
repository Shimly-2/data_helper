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

@DATA_PROCESSER_REGISTRY.register("GenMetaINFO")
class genmetaINFO(BaseDataProcessor):
    """生产meta数据，将所有视频，图片路径变为绝对路径"""
    def __init__(
        self,
        workspace,
        json_root,
        dataset_root,
        task_name,
        case_name,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.task_name = task_name
        self.case_name = case_name
        self.json_root = json_root
        self.dataset_root = dataset_root
        self.timestamp_maker = RobotTimestampsIncoder()

    def gen_per_meta(self, meta_info, json_path):
        with open(json_path,"r") as f:
            meta_info = json.load(f)
        print(f"PID[{os.getpid()}: Load json file - {json_path}")
        case_name = meta_info["case_info"]["case_name"]
        task_name = meta_info["case_info"]["task_name"]
        assert case_name == self.case_name, "Not the same case: meta-{}/config-{}".format(case_name, self.case_name)
        assert task_name == self.task_name, "Not the same task: meta-{}/config-{}".format(task_name, self.task_name)
        root_path = os.path.join(self.dataset_root, task_name, case_name)
        frame_infos = meta_info["frame_info"]
        for idx in frame_infos:
            frame_info = frame_infos[idx]
            for cam_view in frame_info["cam_views"]:
                view_info = frame_info["cam_views"][cam_view]
                view_info["rgb_video_path"] = os.path.join(root_path, view_info["rgb_video_path"])
                view_info["rgb_img_path"] = os.path.join(root_path, view_info["rgb_img_path"])
            for tactile_view in frame_info["tactile_views"]:
                view_info = frame_info["tactile_views"][tactile_view]
                view_info["rgb_video_path"] = os.path.join(root_path, view_info["rgb_video_path"])
                view_info["rgb_img_path"] = os.path.join(root_path, view_info["rgb_img_path"])
        return meta_info
        
    def process(self, meta, task_infos):
        results = []
        json_paths = []
        self.json_root = os.path.join(self.json_root, self.task_name, self.case_name)
        for dirpath, dirnames, filenames in os.walk(self.json_root):
            for filename in filenames:
                if filename.endswith(".json"):
                    json_paths.append(os.path.join(dirpath, filename))
        if self.pool > 1:
            args_list = []
            meta_info = []
            for json_path in json_paths:  #依次读取视频文件
                args_list.append((meta_info, json_path))
            results = self.multiprocess_run(self.gen_per_meta, args_list)
        else:
            for idx, json_path in enumerate(json_paths):  #依次读取视频文件
                filename = osp.split(json_path)[-1]
                # save_new_filename = osp.join(self.save_root, filename)
                self.logger.info(
                    f"Start process {idx+1}/{len(json_paths)}")
                results.append(
                    self.gen_per_meta(json_path))
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            meta.append(meta_infos)
        return meta, task_infos
