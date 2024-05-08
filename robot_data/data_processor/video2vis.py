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
import matplotlib.pyplot as plt
import PIL.Image as Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

@DATA_PROCESSER_REGISTRY.register("Video2Vis")
class video2vis(BaseDataProcessor):
    """将结果可视化"""
    def __init__(
        self,
        workspace,
        dataset_root,
        task_name,
        case_name,
        cam_views,
        tactile_views,
        fps,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.task_name = task_name
        self.case_name = case_name
        self.cam_views = cam_views
        self.tactile_views = tactile_views
        self.dataset_root = dataset_root
        self.fps = fps
        self.timestamp_maker = RobotTimestampsIncoder()

    def plot2D(self,force_com):
        plt.ion()
        self.x.append(force_com[0])
        self.y.append(force_com[1])
        self.z.append(force_com[2])
        self.num.append(self.timestep)
        self.timestep = self.timestep + 1
        self.ax2D.cla()#清除当前Axes对象
        self.ax2D.plot(self.num, self.x, label='x')
        self.ax2D.plot(self.num, self.y, label='y')
        self.ax2D.plot(self.num, self.z, label='z')
        # ax.set_xlim3d(0, 20)  # 指定x轴坐标值范围
        # ax.set_ylim3d(0, 20)  # 指定y轴坐标值范围
        # ax.set_zlim3d(0, 50)  # 指定z轴坐标值范围
        plt.show()
        plt.pause(0.01)
        canvas = FigureCanvasAgg(plt.gcf())
        # 绘制图像
        canvas.draw()
        # 获取图像尺寸
        w, h = canvas.get_width_height()
        # 解码string 得到argb图像
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        # 重构成w h 4(argb)图像
        buf.shape = (w, h, 4)
        # 转换为 RGBA
        buf = np.roll(buf, 3, axis=2)
        # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        # 转换为numpy array rgba四通道数组
        image = np.asarray(image)
        # 转换为rgb图像
        rgb_image = image[:, :, :3]
        # 转换为bgr图像
        r,g,b=cv2.split(rgb_image)
        img_bgr = cv2.merge([b,g,r])
        '''生成视频'''
        self.videoWriter.write(img_bgr)
    
    def plot3D(self,position_):
        plt.ion()
        self.x_p.append(position_[0])
        self.y_p.append(position_[1])
        self.z_p.append(position_[2])
        self.ax3Dpos.plot(self.x_p, self.y_p, self.z_p, "gx--")  # 参数与二维折现图不同的在于多了一个Z轴的数据
        self.ax3Dpos.set_xlim3d(0, 0.7)  # 指定x轴坐标值范围
        self.ax3Dpos.set_ylim3d(-0.5, 0.5)  # 指定y轴坐标值范围
        self.ax3Dpos.set_zlim3d(0.645, 0.8)  # 指定z轴坐标值范围
        plt.show()
        plt.pause(0.01)

    def vis_per_meta(self, meta_info, object_name, epoch, fps):
        case_name = meta_info["case_info"]["case_name"]
        task_name = meta_info["case_info"]["task_name"]
        assert case_name == self.case_name, "Not the same case: meta-{}/config-{}".format(case_name, self.case_name)
        assert task_name == self.task_name, "Not the same task: meta-{}/config-{}".format(task_name, self.task_name)
        vis_path = os.path.join(self.dataset_root, task_name, case_name, "vis_meta", object_name, f"{epoch}.mp4")
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # TODO change to the meta
        img_size = [320, 256]
        video_recorder = cv2.VideoWriter(vis_path, fourcc, fps, (img_size[0] * 3, img_size[1] * 2), True)
        frame_infos = meta_info["frame_info"]

        fig3Dpos = plt.figure(figsize=(10,8), dpi=320)
        ax3Dpos = Axes3D(fig3Dpos, auto_add_to_figure=False)  #实例化Axes3D对象，创建3D图像（注意：见下方注释）
        fig3Dpos.add_axes(ax3Dpos) # 手动将3D图像添加到画布对象上
        fig2D = plt.figure(figsize=(10,8), dpi=320)
        ax2D = fig2D.add_subplot(1, 1, 1)

        pbar = tqdm(total=meta_info["case_info"]["steps"], colour='green', desc=f'PID[{os.getpid()}]: Vis<{epoch}-{object_name}>')
        for idx in frame_infos:
            frame_info = frame_infos[idx]
            imgs = dict()
            result_img = []
            for cam_view in self.cam_views:
                view_info = frame_info["cam_views"][cam_view]
                imgs[cam_view] = cv2.imread(view_info["rgb_img_path"])
            for tactile_view in frame_info["tactile_views"]:
                view_info = frame_info["tactile_views"][tactile_view]
                imgs[tactile_view] = cv2.imread(view_info["rgb_img_path"])
            target_position = frame_info["tar_object_states"]["tar_object_position"]
            target_orientation = frame_info["tar_object_states"]["tar_object_orientation"]
            ee_cmd_position = frame_info["commended"]["ee_command_position"]
            ee_cmd_orientation = frame_info["commended"]["ee_command_orientation"]
            gripper_cmd = frame_info["commended"]["gripper_closedness_commanded"]
            dis_to_target = frame_info["distance_to_target"]
            normalForceL = frame_info["normalForceL"]
            normalForceR = frame_info["normalForceR"]
            lateralForceL = frame_info["lateralForceL"]
            lateralForceR = frame_info["lateralForceR"]
            is_success = frame_info["success"]

            # data_img = np.ones((imgs["front"].shape[0], imgs["front"].shape[1], 3), np.uint8) * 255
            cv2.putText(
                img=imgs["front"], 
                text=("Step: "+ str(idx)+ " dis: "+ str(round(dis_to_target, 3))), 
                org=(0, 20),
                fontFace=cv2.FONT_ITALIC, 
                fontScale=0.75, 
                color=(0, 0, 0), 
                thickness=2
            )
            cv2.putText(
                img=imgs["front"], 
                text=("Gripper: "+ str(round(gripper_cmd, 3))), 
                org=(0, 40),
                fontFace=cv2.FONT_ITALIC, 
                fontScale=0.75, 
                color=(0, 0, 0), 
                thickness=2
            )
            cv2.putText(
                img=imgs["front"], 
                text=("Success"), 
                org=(0, 60),
                fontFace=cv2.FONT_ITALIC, 
                fontScale=0.75, 
                color=(0, 0, 255) if not is_success else (0, 255, 0), 
                thickness=2
            )
            cv2.putText(
                img=imgs["gelsightL"], 
                text=("norForceL: "+ str(round(normalForceL, 3))), 
                org=(0, 20),
                fontFace=cv2.FONT_ITALIC, 
                fontScale=0.75, 
                color=(0, 0, 0), 
                thickness=2
            )
            shear_list = ["shearX", "shearY", "shearZ"]
            shear_idx = 0
            for shear in shear_list:
                cv2.putText(
                    img=imgs["gelsightL"], 
                    text=(f"{shear}: {round(lateralForceL[shear_idx], 3)}"), 
                    org=(0, 20 *(shear_idx+2)),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=0.75, 
                    color=(0, 0, 0), 
                    thickness=2
                )
                shear_idx = shear_idx + 1
            cv2.putText(
                img=imgs["gelsightR"], 
                text=("norForceR: "+ str(round(normalForceR, 3))), 
                org=(0, 20),
                fontFace=cv2.FONT_ITALIC, 
                fontScale=0.75, 
                color=(0, 0, 0), 
                thickness=2
            )
            shear_idx = 0
            for shear in shear_list:
                cv2.putText(
                    img=imgs["gelsightR"], 
                    text=(f"{shear}: {round(lateralForceR[shear_idx], 3)}"), 
                    org=(0, 20 *(shear_idx+2)),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=0.75, 
                    color=(0, 0, 0), 
                    thickness=2
                )
                shear_idx = shear_idx + 1
            h1 = np.concatenate((imgs["front"], imgs["gelsightL"], imgs["gelsightR"]), axis=1)
            h2 = np.concatenate((imgs["side"], imgs["wrist"], imgs["fronttop"]), axis=1)
            result_img = np.concatenate((h1, h2), axis=0)
            video_recorder.write(result_img)
            pbar.update(1)
        video_recorder.release()
        self.logger.info(f"Success visualized in - {vis_path}")
        return meta_info
        
    def process(self, meta, task_infos):
        results = []
        args_list = []
        if self.pool > 1:
            for meta_info in meta:  #依次读取视频文件
                object_name = meta_info["case_info"]["object_name"]
                epoch = meta_info["case_info"]["epoch"]
                args_list.append((meta_info, object_name, epoch, self.fps))
            results = self.multiprocess_run(self.vis_per_meta, args_list)
        else:
            for idx, meta_info in enumerate(meta):  #依次读取视频文件
                self.logger.info(
                    f"Start process {idx+1}/{len(meta)}")
                results.append(
                    self.vis_per_meta(meta_info, object_name, epoch, self.fps))
        # for meta_infos in results:
        #     # TODO 试试看不写这句行不行
        #     meta.append(meta_infos)
        return meta, task_infos
