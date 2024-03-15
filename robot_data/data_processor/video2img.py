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

@DATA_PROCESSER_REGISTRY.register("Video2Img")
class video2img(BaseDataProcessor):

    def __init__(
        self,
        workspace,
        video_root,
        new_img_save_root,
        task_name,
        case_name,
        save_framerate=3,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.save_framerate = save_framerate
        self.task_name = task_name
        self.case_name = case_name
        self.video_root = video_root
        self.save_root = new_img_save_root

        self.timestamp_maker = RobotTimestampsIncoder()

    def split_per_video(self, video_path, save_path, save_framerate):
        os.makedirs(save_path, exist_ok=True)
        object_name = video_path.split("/")[-3]
        epoch = video_path.split("/")[-2]
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
        for i in trange(int(cap_num/save_framerate), colour='green', desc=f'PID[{os.getpid()}]: Split<{epoch}-{object_name}-{view}>'):
            rval, frame = vc.read()
            # cnt += 1
            if rval:
                if i % save_framerate == 0:
                    timestamp = self.timestamp_maker.set_current_timestamp()
                    # savename = os.path.join(save_path, timestamp + '.jpg')
                    savename = os.path.join(save_path, f'rgb_{i}.jpg')
                    cv2.imwrite(savename, frame)
                    # self.logger.info(f"Start split {int(cnt/save_framerate)}/{int(cap_num/save_framerate)}, savename - {savename}")
            else:
                break
        vc.release()
        self.logger.info(f'Done: {video_path}')
        
    def process(self, meta, task_infos):
        results = []
        video_paths = []
        for dirpath, dirnames, filenames in os.walk(osp.join(self.video_root, self.task_name, self.case_name)):
            for filename in filenames:
                if filename.endswith(".mp4"):
                    video_paths.append(os.path.join(dirpath, filename))
        if self.pool > 1:
            args_list = []
            for video_path in video_paths:  #依次读取视频文件
                object_name = video_path.split("/")[-3]
                epoch = video_path.split("/")[-2]
                view = video_path.split("/")[-1].split(".")[0]
                # filename = f ilename.split("#")[0]
                save_new_filename = osp.join(self.save_root, self.task_name, self.case_name, "train_metas", object_name, view, epoch, "rgb")
                args_list.append((video_path, save_new_filename, self.save_framerate))
            results = self.multiprocess_run(self.split_per_video, args_list)
        else:
            for idx, video_path in enumerate(video_paths):  #依次读取视频文件
                filename = osp.split(video_path)[-1]
                save_new_filename = osp.join(self.save_root, filename)
                self.logger.info(
                    f"Start process {idx+1}/{len(video_paths)}")
                results.append(
                    self.split_per_video(video_path, save_new_filename, self.save_framerate))
        return meta, task_infos
