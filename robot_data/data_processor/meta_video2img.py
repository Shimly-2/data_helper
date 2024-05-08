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

@DATA_PROCESSER_REGISTRY.register("MetaVideo2Img")
class meta_video2img(BaseDataProcessor):

    def __init__(
        self,
        workspace,
        video_root,
        save_framerate=3,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.save_framerate = save_framerate
        self.video_root = video_root

        self.timestamp_maker = RobotTimestampsIncoder()

    def gen_video_path_list(self, frame_info):
        video_path_list = []
        for cam_view in frame_info["cam_views"]:
            view_info = frame_info["cam_views"][cam_view]
            video_path_list.append(view_info["rgb_video_path"])
            os.makedirs(os.path.dirname(view_info["rgb_img_path"]), exist_ok=True)
        for tactile_view in frame_info["tactile_views"]:
            view_info = frame_info["tactile_views"][tactile_view]
            video_path_list.append(view_info["rgb_video_path"])
            os.makedirs(os.path.dirname(view_info["rgb_img_path"]), exist_ok=True)
        return video_path_list
    
    def find_value(self, dictionary, target_key):
        # 遍历字典的所有项
        for key, value in dictionary.items():
            # 如果当前项的 key 等于目标 key，返回对应的 value
            if key == target_key:
                return value
            # 如果当前项的 value 是字典类型，则递归搜索
            elif isinstance(value, dict):
                result = self.find_value(value, target_key)
                if result is not None:
                    return result
        # 如果未找到目标 key，返回 None
        return None
    
    def split_per_video(self, meta_info, save_framerate):
        video_path_list = self.gen_video_path_list(meta_info["frame_info"][str(0)])
        steps = meta_info["case_info"]["steps"]
        epoch = meta_info["case_info"]["epoch"]
        object_name = meta_info["case_info"]["object_name"]
        for video_path in video_path_list:
            vc = cv2.VideoCapture(video_path)
            view = video_path.split("/")[-1].split(".")[0]
            cap_num = int(vc.get(7))
            assert cap_num == meta_info["case_info"]["steps"], f"The video {video_path} does't match meta file, [cap_num]--<{cap_num}>, [meta]--<{steps}>"
            cap_width = math.ceil(vc.get(3))
            cap_height = math.ceil(vc.get(4))
            # assert cap_width==3840 and cap_height==1920
            rval = vc.isOpened()
            assert rval, "video path not exist - {}".format(video_path)
            for i in trange(cap_num, colour='green', desc=f'PID[{os.getpid()}]: Split<{epoch}-{object_name}-{view}>'):
                rval, frame = vc.read()
                # cnt += 1
                if rval:
                    if i % save_framerate == 0:
                        timestamp = self.timestamp_maker.set_current_timestamp()
                        savename = self.find_value(meta_info["frame_info"][str(i)], view)
                        cv2.imwrite(savename["rgb_img_path"], frame)
                else:
                    break
            vc.release()
            self.logger.info(f'Done: {video_path}')
        
    def process(self, meta, task_infos):
        results = []
        if self.pool > 1:
            args_list = []
            for meta_info in meta:  #依次读取视频文件
                args_list.append((meta_info, self.save_framerate))
            results = self.multiprocess_run(self.split_per_video, args_list)
        else:
            for idx, meta_info in enumerate(meta):  #依次读取视频文件
                self.logger.info(
                    f"Start process {idx+1}/{len(meta)}")
                results.append(
                    self.split_per_video(meta_info, self.save_framerate))
        return meta, task_infos
