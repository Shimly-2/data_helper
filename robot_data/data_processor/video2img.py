from .BaseDataProcessor import BaseDataProcessor
import cv2
import time
import os
from tqdm import tqdm
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
        save_framerate=3,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.save_framerate = save_framerate
        self.video_root = video_root
        self.save_root = new_img_save_root
        # self.new_gt_save_dir = new_gt_save_dir
        os.makedirs(self.save_root, exist_ok=True)

        self.timestamp_maker = RobotTimestampsIncoder()
        # os.makedirs(new_gt_save_dir, exist_ok=True)

    def split_per_video(self, video_path, save_path, save_framerate):
        # save_dir = save_path + '/' + newname
        vc = cv2.VideoCapture(osp.join(self.video_root, video_path))
        cap_num = int(vc.get(7))
        cap_width = math.ceil(vc.get(3))
        cap_height = math.ceil(vc.get(4))
        # assert cap_width==3840 and cap_height==1920
        rval = vc.isOpened()
        assert rval, "video path not exist - {}".format(osp.join(self.video_root, video_path))
        cnt = -1
        while rval:
            rval, frame = vc.read()
            cnt += 1
            if rval:
                if cnt % save_framerate == 0:
                    timestamp = self.timestamp_maker.set_current_timestamp()
                    savename = os.path.join(save_path, timestamp + '.jpg')
                    # cv2.imwrite(savename, frame)
                    self.logger.info(f"Start split {int(cnt/3)}/{int(cap_num/3)}, savename - {savename}")
            else:
                break
        vc.release()
        print('Done : ' + str() + video_path)
        
    def process(self, meta, task_infos):
        results = []
        # os.makedirs(osp.join(self.save_root, 'camera_name'),
        #             exist_ok=True)
        if self.pool > 1:
            args_list = []
            video_paths = os.listdir(self.video_root) #返回指定路径下的文件和文件夹列表。
            for video_path in video_paths:  #依次读取视频文件
            #for meta_file, ceph_dir in zip(self.meta_files, self.ceph_dirs):
                filename = osp.split(video_path)[-1]
                filename = filename.split("#")[0]
                save_new_filename = osp.join(self.save_root, filename)
                # save_fixed_filename = osp.join(self.save_root, 'camera_name', filename)
                args_list.append((video_path, save_new_filename, self.save_framerate))
            results = self.multiprocess_run(self.split_per_video, args_list)
        else:
            video_paths = os.listdir(self.video_root) #返回指定路径下的文件和文件夹列表。
            for idx, video_path in enumerate(video_paths):  #依次读取视频文件
            # for idx, (meta_file, ceph_dir) in enumerate(
            #         zip(self.meta_files, self.ceph_dirs)):
                filename = osp.split(video_path)[-1]
                save_new_filename = osp.join(self.save_root, filename)
                # save_fixed_filename = osp.join(self.save_root,
                #                                'fix_unmatched_solid', filename)
                self.logger.info(
                    f"start process {idx+1}/{len(video_paths)}")
                results.append(
                    self.split_per_video(video_path, save_new_filename, self.save_framerate))
        # filtered_all = 0
        # for filtered_nums in results:
        #     filtered_all += filtered_nums
        # self.logger.info(f'filter {filtered_all} frames')
        # task_infos.update({
        #     'filter_nums': filtered_all,
        #     'new_meta_save_root': self.save_root
        # })
        return meta, task_infos
