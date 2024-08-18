from .BaseDataProcessor import BaseDataProcessor
import cv2
import time
import os
from tqdm import tqdm
from tqdm import trange
import math
import multiprocessing
import os.path as osp
import numpy as np
import copy
from robot_data.utils.registry_factory import DATA_PROCESSER_REGISTRY
from robot_data.utils.robot_timestamp import RobotTimestampsIncoder

@DATA_PROCESSER_REGISTRY.register("Img2Video")
class Img2Video(BaseDataProcessor):

    def __init__(
        self,
        workspace,
        dataset_root,
        views,
        fps=20,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.fps = fps
        self.dataset_root = dataset_root
        self.views = views
        self.timestamp_maker = RobotTimestampsIncoder()

    def synthesis_per_video(self, raw_img_list, save_root, fourcc):
        os.makedirs(save_root, exist_ok=True)
        object_name = save_root.split("/")[-3]
        for view in self.views:
            save_path = os.path.join(save_root, view + f"#{self.fps}.mp4")
            # if os.path.exists(save_path):
            #     continue
            if not raw_img_list[0].endswith(".jpg"):
                ref_img_path = raw_img_list[0] + ".jpg"
            else:
                ref_img_path = raw_img_list[0]
            ref_img_path = ref_img_path.replace("rgb", view)
            ref_img = cv2.imread(ref_img_path)
            videoWriter = cv2.VideoWriter(save_path, fourcc, self.fps, ref_img.shape[1::-1])
            for raw_img_path in tqdm(raw_img_list, colour="green", desc=f"PID[{os.getpid()}]: Process<{object_name}--{view}>"):
                if not raw_img_path.endswith(".jpg"):
                    raw_img_path = raw_img_path + ".jpg"
                view_img_path = raw_img_path.replace("rgb", view)
                view_img = cv2.imread(view_img_path)
                videoWriter.write(view_img)
            videoWriter.release()
            self.logger.info(f'Done: {save_path}')

    def gen_meta_infos(self, meta):
        video_paths = []
        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in filenames:
                if filename.endswith(".mp4") and filename.find("#") < 0:
                    video_paths.append(os.path.join(dirpath, filename))
        for video_path in video_paths:  #依次读取视频文件
            '''/root/raw_meta/xxx.mp4-->/root/rgb/xxx/'''
            save_new_root = video_path.replace("raw_meta", "rgb").split(".")[0]
            meta.append({"raw_meta_root": save_new_root})
        return meta
        
    def process(self, meta, task_infos):
        if isinstance(meta, list):
            if len(meta) == 0:
                meta = self.gen_meta_infos(meta)
        results = []
        if self.pool > 1:
            args_list = []
            for per_meta in meta:
                raw_img_list = os.listdir(per_meta["raw_meta_root"]) 
                '''sort rgb_1.jpg/xxx_1.jpg'''
                raw_img_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-1])) 
                raw_img_list = [os.path.join(per_meta["raw_meta_root"], x) for x in raw_img_list]
                task_path = os.path.dirname(raw_img_list[0]).replace("rgb", "final_meta")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                args_list.append((raw_img_list, task_path, fourcc))
            results = self.multiprocess_run(self.synthesis_per_video, args_list)
        else:
            for idx, per_meta in enumerate(meta):
                raw_img_list = os.listdir(per_meta["raw_meta_root"]) 
                '''sort rgb_1.jpg/xxx_1.jpg'''
                raw_img_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-1])) 
                raw_img_list = [os.path.join(per_meta["raw_meta_root"], x) for x in raw_img_list]
                task_path = os.path.dirname(raw_img_list[0]).replace("rgb", "final_meta")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.logger.info(
                    f"Start process {idx+1}/{len(meta)}")
                results.append(
                    self.synthesis_per_video(raw_img_list, task_path, fourcc))
        return meta, task_infos