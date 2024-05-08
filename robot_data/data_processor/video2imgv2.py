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

@DATA_PROCESSER_REGISTRY.register("Video2ImgV2")
class video2imgV2(BaseDataProcessor):

    def __init__(
        self,
        workspace,
        video_root,
        save_framerate=1,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.save_framerate = save_framerate
        self.video_root = video_root
        self.timestamp_maker = RobotTimestampsIncoder()

    def split_per_video(self, video_path, save_root, save_framerate):
        os.makedirs(save_root, exist_ok=True)
        object_name = video_path.split("/")[-3]
        view = video_path.split("/")[-1].split(".")[0]
        vc = cv2.VideoCapture(video_path)
        cap_num = int(vc.get(7))
        cap_width = math.ceil(vc.get(3))
        cap_height = math.ceil(vc.get(4))
        # assert cap_width==3840 and cap_height==1920
        rval = vc.isOpened()
        assert rval, "video path not exist - {}".format(video_path)
        cnt = -1
        # while rval:
        for i in trange(int(cap_num/save_framerate), colour='green', desc=f'PID[{os.getpid()}]: Split<{object_name}--{view}>'):
            rval, frame = vc.read()
            cnt += 1
            if rval:
                if i % save_framerate == 0:
                    savename = os.path.join(save_root, f'rgb_{i}.jpg')
                    cv2.imwrite(savename, frame)
                    # self.logger.info(f"Start split {int(cnt/save_framerate)}/{int(cap_num/save_framerate)}, savename - {savename}")
            else:
                break
            # if cnt==30:
            #     break
        vc.release()
        self.logger.info(f'Done: {video_path}')
        
    def process(self, meta, task_infos):
        # meta = dict()
        # meta["raw_meta_root"] = []
        results = []
        video_paths = []
        for dirpath, dirnames, filenames in os.walk(self.video_root):
            for filename in filenames:
                if filename.endswith(".mp4") and filename.find("#") < 0:
                    video_paths.append(os.path.join(dirpath, filename))
        if self.pool > 1:
            args_list = []
            for video_path in video_paths:  #依次读取视频文件
                '''/root/raw_meta/xxx.mp4-->/root/rgb/xxx/'''
                save_new_root = video_path.replace("raw_meta", "rgb").split(".")[0]
                meta.append({"raw_meta_root": save_new_root})
                args_list.append((video_path, save_new_root, self.save_framerate))
            results = self.multiprocess_run(self.split_per_video, args_list)
        else:
            for idx, video_path in enumerate(video_paths):  #依次读取视频文件
                filename = osp.split(video_path)[-1]
                '''/root/raw_meta/xxx.mp4-->/root/rgb/xxx/'''
                save_new_root = video_path.replace("raw_meta", "rgb").split(".")[0]
                meta.append({"raw_meta_root": save_new_root})
                self.logger.info(
                    f"Start process {idx+1}/{len(video_paths)}")
                results.append(
                    self.split_per_video(video_path, save_new_root, self.save_framerate))
        return meta, task_infos
