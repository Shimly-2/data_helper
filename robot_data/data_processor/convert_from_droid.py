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

@DATA_PROCESSER_REGISTRY.register("ConvertFromDROID")
class ConvertFromDROID(BaseDataProcessor):
    """根据window set生成训练集和验证集, 无多线程"""
    def __init__(
        self,
        workspace,
        dataset_root,
        split_set,
        img_size,
        fps,
        dataset_key,
        random_seed,
        obs_image,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.split_set = split_set
        self.random_seed = random_seed
        self.dataset_key = dataset_key
        self.obs_image = obs_image
        self.img_size = img_size
        self.fps = fps
        self.timestamp_maker = RobotTimestampsIncoder()      

    def convert_results(self, e_idx, episode, root_path):
        meta_info = []
        results = dict()
        results["case_info"] = dict()
        results["case_info"]["img_size"] = self.img_size
        results["case_info"]["epoch"] = e_idx
        results["case_info"]["task_name"] = "droid_dataset"
        results["case_info"]["case_name"] = "droid_100"
        results["case_info"]["object_name"] = self.split_set
        results["case_info"]["fps"] = self.fps
        results["frame_info"] = dict()
        for idx, step in enumerate(tqdm(episode["steps"], colour='green', desc=f'Convert Results[e-{e_idx}]')):
            results["frame_info"][idx] = dict()
            results["frame_info"][idx] = dict()
            results["frame_info"][idx]["cam_views"] = dict()
            for image_type in self.obs_image:
                results["frame_info"][idx]["cam_views"][image_type] = dict()
                results["frame_info"][idx]["cam_views"][image_type]["rgb_video_path"] = os.path.join(root_path, f"raw_meta/{self.split_set}/epoch_{e_idx}/{image_type}.mp4")
            results["frame_info"][idx]["commended"] = dict()
            results["frame_info"][idx]["commended"]["gripper_closedness_commanded"] = list(step["observation"]["gripper_position"].numpy())[0]
            results["frame_info"][idx]["commended"]["ee_command_position"] = list(step["observation"]["cartesian_position"].numpy()[:3])
            results["frame_info"][idx]["commended"]["ee_command_orientation"] = list(step["observation"]["cartesian_position"].numpy()[3:])
            results["frame_info"][idx]["jointstates"] = dict2list(step["observation"]["joint_position"].numpy())
            results["frame_info"][idx]["instruction"] = step["language_instruction"].numpy().decode('utf-8')
            if results["frame_info"][idx]["instruction"] == '':
                results["frame_info"][idx]["instruction"] = step["language_instruction_2"].numpy().decode('utf-8')
            if results["frame_info"][idx]["instruction"] == '':
                results["frame_info"][idx]["instruction"] = step["language_instruction_3"].numpy().decode('utf-8')
        results["case_info"]["steps"] = len(results["frame_info"])
        with open(os.path.join(root_path, f"raw_meta/{self.split_set}/epoch_{e_idx}/result.json"), "w") as f:
            json.dump(results, f)
        return meta_info
    
    def convert_videos(self, e_idx, episode, root_path):
        videoWriter = dict()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for image_type in self.obs_image:
            os.makedirs(os.path.join(root_path, f"raw_meta/{self.split_set}/epoch_{e_idx}"), exist_ok=True)
            videoWriter[image_type] = cv2.VideoWriter(os.path.join(root_path, f"raw_meta/{self.split_set}/epoch_{e_idx}/{image_type}.mp4"), fourcc, self.fps, self.img_size)
        for i, step in enumerate(tqdm(episode["steps"], colour='green', desc=f'Convert Videos[e-{e_idx}]')):
            for image_type in self.obs_image:
                img = step["observation"][image_type].numpy()
                videoWriter[image_type].write(img)
        for image_type in self.obs_image:
            videoWriter[image_type].release()
        
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
                self.convert_videos(e_idx, episode, root_path)
                self.convert_results(e_idx, episode, root_path)
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            meta.append(meta_infos)
        return meta, task_infos
