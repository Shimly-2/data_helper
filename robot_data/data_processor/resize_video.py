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
import glob
import ffmpeg

@DATA_PROCESSER_REGISTRY.register("ResizeVideo")
class ResizeVideo(BaseDataProcessor):

    def __init__(
        self,
        workspace,
        dataset_root,
        target_cam_view,
        target_tactile_view,
        aw: float=0.5,
        ah: float=0.5,
        fps: int=10,
        size: int=224,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.aw = aw
        self.ah = ah
        self.fps = fps
        self.size = size
        self.target_cam_view = target_cam_view
        self.target_tactile_view = target_tactile_view
        self.timestamp_maker = RobotTimestampsIncoder()

    def preprocess_videos(self, video_path, output_dir):
        name_clip = video_path.split("/")[-1].replace(".mp4", "")
        # Initialize paths.
        processed_video_path = os.path.join(output_dir, name_clip + "#s224.mp4")
        raw_video_path = video_path

        if not os.path.exists(processed_video_path):
            cap = cv2.VideoCapture(raw_video_path)

            # Get the frame width and height
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate crop dimensions
            crop_x = int((frame_width - min(frame_width, frame_height)) * self.aw)
            crop_y = int((frame_height - min(frame_width, frame_height)) * self.ah)
            crop_width = crop_height = min(frame_width, frame_height)

            # Initialize VideoWriter object
            out = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.size, self.size))

            # Read and process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Crop the frame
                cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
                # Resize the frame
                resized_frame = cv2.resize(cropped_frame, (self.size, self.size))
                # Write the frame to output video
                out.write(resized_frame)
            # Release VideoCapture and VideoWriter objects
            cap.release()
            out.release()

    def process(self, meta, task_infos):
        # meta = dict()
        # meta["raw_meta_root"] = []
        results = []
        video_paths = []
        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in filenames:
                if filename.endswith(".mp4") and filename.find("#") < 0:
                    if filename.find(self.target_cam_view) >= 0 or filename.find(self.target_tactile_view) >= 0:
                        video_paths.append(os.path.join(dirpath, filename))
        if self.pool > 1:
            args_list = []
            for video_path in video_paths:  #依次读取视频文件
                meta.append({"raw_meta_root": video_path})
                output_dir = os.path.dirname(video_path)
                args_list.append((video_path, output_dir))
            results = self.multiprocess_run(self.preprocess_videos, args_list)
        else:
            for idx, video_path in enumerate(video_paths):  #依次读取视频文件
                meta.append({"raw_meta_root": video_path})
                output_dir = os.path.dirname(video_path)
                self.logger.info(
                    f"Start process {idx+1}/{len(video_paths)}")
                results.append(
                    self.preprocess_videos(video_path, output_dir))
        return meta, task_infos
