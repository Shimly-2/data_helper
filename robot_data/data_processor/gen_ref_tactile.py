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

@DATA_PROCESSER_REGISTRY.register("Gen_Ref_Tactile")
class gen_ref_tactile(BaseDataProcessor):

    def __init__(
        self,
        workspace,
        video_root,
        ref_img_save_root,
        ref_img_name,
        sample_num=30,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.video_root = video_root
        self.save_root = ref_img_save_root
        self.ref_img_name = ref_img_name
        self.sample_num = sample_num

        self.timestamp_maker = RobotTimestampsIncoder()
    
    def generate_ref_img(self, video_name, sample_num):
        frame_avg = []
        vc = cv2.VideoCapture(osp.join(self.video_root, video_name)) #读入视频文件
        fps = vc.get(cv2.CAP_PROP_FPS) #获取帧率
        ret = vc.isOpened() #判断视频是否打开 返回True或Flase
        frame_sum = np.zeros((240,320,3)).astype(np.float64)
        count = 1

        while ret and count<=sample_num: #循环读取视频帧
            rval, frame = vc.read() #videoCapture.read() 函数，第一个返回值为是否成功获取视频帧，第二个返回值为返回的视频帧：
            count = count + 1
            frame_sum += frame
            
        frame_temp = frame_sum / sample_num
        frame_avg.append(frame_temp.astype(np.uint8))
        means, std = cv2.meanStdDev(frame_avg[-1])
        self.logger.info(f"Ref img update ---> means: {means}, std: {std}")
        ref_img = np.zeros((240,320,3)).astype(np.float64)
        for i in range(len(frame_avg)):
            ref_img += frame_avg[i]
        ref_img = ref_img / len(frame_avg)
        ref_img = ref_img.astype(np.uint8)
        return ref_img
        # cv2.imwrite(ref_img_path, ref_img) #存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
        # self.logger.info(f"Ref img is generated successfully in {ref_img_path}")
        # cv2.waitKey(1) #waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下 键,则接续等待(循环)
        
    def process(self, meta, task_infos):
        os.makedirs(self.save_root, exist_ok=True)
        results = []
        if self.pool > 1:
            args_list = []
            video_names = os.listdir(self.video_root) #返回指定路径下的文件和文件夹列表。
            for video_name in video_names:  #依次读取视频文件
                if(video_name.find(".mp4")<0):
                    continue
                # save_fixed_filename = osp.join(self.save_root, 'camera_name', filename)
                args_list.append((video_name, self.sample_num))
            results = self.multiprocess_run(self.generate_ref_img, args_list)
        else:
            video_names = os.listdir(self.video_root) #返回指定路径下的文件和文件夹列表。
            for idx, video_name in enumerate(video_names):  #依次读取视频文件
                if(video_name.find(".mp4")<0):
                    continue
            # for idx, (meta_file, ceph_dir) in enumerate(
                # save_fixed_filename = osp.join(self.save_root,
                #                                'fix_unmatched_solid', filename)
                self.logger.info(
                    f"start process {idx+1}/{len(video_names)}")
                results.append(
                    self.generate_ref_img(video_name, self.sample_num))
        ref_img = np.zeros((240,320,3)).astype(np.float64)
        for i in range(len(results)):
            ref_img += results[i]
        ref_img = ref_img / len(results)
        ref_img = ref_img.astype(np.uint8)
        ref_img_path = osp.join(self.save_root, self.ref_img_name)
        cv2.imwrite(ref_img_path, ref_img) #存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
        self.logger.info(f"Ref img is generated successfully in {ref_img_path}")
        return meta, task_infos
