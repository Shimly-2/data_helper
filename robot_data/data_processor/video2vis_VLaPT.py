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
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import textwrap

@DATA_PROCESSER_REGISTRY.register("Video2VisVLaPT")
class Video2VisVLaPT(BaseDataProcessor):
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
        self.color_map = {
            "action_stage": (225, 225, 255), 
            "grasp_label": (159, 217, 192),
            "object": (222, 235, 247), 
            "material": (255, 242, 204), 
            "texture": (251, 229, 214), 
            "position": (226, 255, 205), 
            "color": (225, 204, 240), 
            "shape": (226, 240 ,217), 
            "surface": (214, 220, 229), 
            "size": (255, 205, 193), 
            "weight": (255, 185, 255), 
            "deformation": (197, 255, 255)
        }
        from collections import deque
        self.language_queue = deque(maxlen=3)

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

    def vis_per_meta(self, meta_info, json_path, fps):
        with open(os.path.join(self.dataset_root, json_path), "r") as f:
            data = json.load(f)
        print(f"PID[{os.getpid()}: Load json file - {json_path}")

        object_name = json_path.split("/")[-3]
        epoch = json_path.split("/")[-2]
        epoch_idx = int(epoch.split("_")[-1])
        if epoch_idx == 0:
            vis_path = os.path.join(self.dataset_root + "_vis", object_name, f"{epoch}.mp4")
            action_vis_path = os.path.join(self.dataset_root + "_vis", object_name, f"{epoch}_action.mp4")
            text_vis_path = os.path.join(self.dataset_root + "_vis", object_name, f"{epoch}_text.mp4")
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # TODO change to the meta
            img_size = [640, 480]
            video_recorder = cv2.VideoWriter(vis_path, fourcc, fps, (img_size[0] * 3, img_size[1] * 2), True)
            action_video_recorder = cv2.VideoWriter(action_vis_path, fourcc, fps, (2400, 2400), True)
            text_video_recorder = cv2.VideoWriter(text_vis_path, fourcc, fps, (2400, 2400), True)
            frame_infos = data["frame_info"]

            fig3Dpos = plt.figure(figsize=(10,8), dpi=320)
            ax3Dpos = Axes3D(fig3Dpos, auto_add_to_figure=False)  #实例化Axes3D对象，创建3D图像（注意：见下方注释）
            fig3Dpos.add_axes(ax3Dpos) # 手动将3D图像添加到画布对象上
            fig2D = plt.figure(figsize=(10,8), dpi=320)
            ax2D = fig2D.add_subplot(1, 1, 1)
            action = []
            language = []
            pbar = tqdm(total=data["case_info"]["steps"], colour='green', desc=f'PID[{os.getpid()}]: Vis<{epoch}-{object_name}>')
            for idx in frame_infos:
                frame_info = frame_infos[idx]
                imgs = dict()
                result_img = []
                for cam_view in self.cam_views:
                    view_info = frame_info["cam_views"][cam_view]
                    imgs[cam_view] = cv2.imread(os.path.join(self.dataset_root, view_info["rgb_img_path"]))
                    imgs[f"{cam_view}_depth"] = np.array(Image.open(os.path.join(self.dataset_root, view_info["depth_img_path"])))
                    # imgs[f"{cam_view}_depth"] = cv2.imread(os.path.join(self.dataset_root, view_info["depth_img_path"]))
                for tactile_view in frame_info["tactile_views"]:
                    view_info = frame_info["tactile_views"][tactile_view]
                    imgs[tactile_view] = cv2.imread(os.path.join(self.dataset_root, view_info["rgb_img_path"]))
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
                is_success = frame_info["grasp_label"] == "holding"
                instruction = frame_info["instruction_roml"]
                cogagent_label = frame_info["cogagent_label"]
                grasp_label = frame_info["grasp_label"]
                action_stage = frame_info["action_stage"]
                cogagent_label["fronttop"]["grasp_label"] = grasp_label
                cogagent_label["fronttop"]["action_stage"] = action_stage
                action.append(ee_cmd_position+ee_cmd_orientation+[gripper_cmd])
                instruction = f"[Step{str(idx)}] " + instruction
                self.language_queue.append(instruction)
                action.append(ee_cmd_position+ee_cmd_orientation+[gripper_cmd])

                # shear_list = ["shearX", "shearY", "shearZ"]
                # shear_idx = 0
                # for shear in shear_list:
                #     cv2.putText(
                #         img=imgs["gelsightL"], 
                #         text=(f"{shear}: {round(lateralForceL[shear_idx], 3)}"), 
                #         org=(0, 20 *(shear_idx+2)),
                #         fontFace=cv2.FONT_ITALIC, 
                #         fontScale=0.75, 
                #         color=(0, 0, 0), 
                #         thickness=2
                #     )
                #     shear_idx = shear_idx + 1
                # cv2.putText(
                #     img=imgs["gelsightR"], 
                #     text=("norForceR: "+ str(round(normalForceR, 3))), 
                #     org=(0, 20),
                #     fontFace=cv2.FONT_ITALIC, 
                #     fontScale=0.75, 
                #     color=(0, 0, 0), 
                #     thickness=2
                # )
                # shear_idx = 0
                # for shear in shear_list:
                #     cv2.putText(
                #         img=imgs["gelsightR"], 
                #         text=(f"{shear}: {round(lateralForceR[shear_idx], 3)}"), 
                #         org=(0, 20 *(shear_idx+2)),
                #         fontFace=cv2.FONT_ITALIC, 
                #         fontScale=0.75, 
                #         color=(0, 0, 0), 
                #         thickness=2
                #     )
                #     shear_idx = shear_idx + 1
                # import pdb;pdb.set_trace()
                depth_normalized = imgs["fronttop_depth"].astype(np.float32)
                depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min())
                colored_depth = cm.jet(depth_normalized)
                colored_depth_rgb = (colored_depth[:, :, :3] * 255).astype(np.uint8)
                imgs["fronttop_depth"] = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)
                # imgs["fronttop_depth"] = np.asanyarray(imgs["fronttop_depth"])
                # imgs["fronttop_depth"] = cv2.applyColorMap(cv2.convertScaleAbs(imgs["fronttop_depth"], alpha=0.03), cv2.COLORMAP_JET)
                # imgs["wrist_depth"] = np.asanyarray(imgs["wrist_depth"])
                # imgs["wrist_depth"] = cv2.applyColorMap(cv2.convertScaleAbs(imgs["wrist_depth"], alpha=0.03), cv2.COLORMAP_JET)
                depth_normalized = imgs["wrist_depth"].astype(np.float32)
                depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min())
                colored_depth = cm.jet(depth_normalized)
                colored_depth_rgb = (colored_depth[:, :, :3] * 255).astype(np.uint8)
                imgs["wrist_depth"] = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)
                # import pdb;pdb.set_trace()
                # imgs["gelsightL"] = cv2.rotate(imgs["gelsightL"], cv2.ROTATE_90_CLOCKWISE)
                imgs["gelsightL"] = cv2.resize(imgs["gelsightL"], (imgs["wrist_depth"].shape[1], imgs["wrist_depth"].shape[0]), interpolation=cv2.INTER_LINEAR)
                # imgs["gelsightR"] = cv2.rotate(imgs["gelsightR"], cv2.ROTATE_90_CLOCKWISE)
                imgs["gelsightR"] = cv2.resize(imgs["gelsightR"], (imgs["wrist_depth"].shape[1], imgs["wrist_depth"].shape[0]), interpolation=cv2.INTER_LINEAR)
                # import pdb;pdb.set_trace()
                cv2.putText(
                    img=imgs["fronttop"], 
                    text=("Fronttop"), 
                    org=(5, 40),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(0, 0, 0), 
                    thickness=3
                )
                cv2.putText(
                    img=imgs["fronttop"], 
                    text=("Step: "+ str(idx)), 
                    org=(5, 80),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(0, 0, 0), 
                    thickness=3
                )
                cv2.putText(
                    img=imgs["fronttop"], 
                    text=("Gripper: "+ str(round(gripper_cmd, 3))), 
                    org=(5, 120),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(0, 0, 0), 
                    thickness=3
                )
                cv2.putText(
                    img=imgs["fronttop"], 
                    text=("Success"), 
                    org=(5, 160),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(0, 0, 255) if not is_success else (0, 255, 0), 
                    thickness=3
                )
                cv2.putText(
                    img=imgs["wrist"], 
                    text=("Wrist"), 
                    org=(5, 40),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(0, 0, 0), 
                    thickness=3
                )
                ee_cmd_position = [round(cmd, 3) for cmd in ee_cmd_position]
                cv2.putText(
                    img=imgs["wrist"], 
                    text=("Pos: "+str(ee_cmd_position)), 
                    org=(5, 80),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(0, 0, 0), 
                    thickness=3
                )
                ee_cmd_orientation = [round(cmd, 3) for cmd in ee_cmd_orientation]
                cv2.putText(
                    img=imgs["wrist"], 
                    text=("Rot: "+ str(ee_cmd_orientation)), 
                    org=(5, 120),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(0, 0, 0), 
                    thickness=3
                )
                cv2.putText(
                    img=imgs["fronttop_depth"], 
                    text=("Fronttop Depth"), 
                    org=(5, 40),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(255, 255, 255), 
                    thickness=3
                )
                cv2.putText(
                    img=imgs["wrist_depth"], 
                    text=("Wrist Depth"), 
                    org=(5, 40),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(255, 255, 255), 
                    thickness=3
                )
                cv2.putText(
                    img=imgs["gelsightL"], 
                    text=("GelsightL"), 
                    org=(5, 40),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(255, 255, 255), 
                    thickness=3
                )
                cv2.putText(
                    img=imgs["gelsightR"], 
                    text=("GelsightR"), 
                    org=(5, 40),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=1.2, 
                    color=(255, 255, 255), 
                    thickness=3
                )

                h1 = np.concatenate((imgs["fronttop"], imgs["fronttop_depth"], imgs["gelsightL"]), axis=1)
                h2 = np.concatenate((imgs["wrist"], imgs["wrist_depth"], imgs["gelsightR"]), axis=1)
                result_img = np.concatenate((h1, h2), axis=0)

                fig = plt.figure(figsize=(8, 8), dpi=300)
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('white')
                cmap = cm.get_cmap("coolwarm")
                ax = self.draw_action(ax, action, weights=[i / (len(action) - 1) for i in range(len(action))], cmap=cmap, color=None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                fig.add_axes(ax)

                fig.subplots_adjust(left=0.0, right=0.9, bottom=-0.1, top=1.3)

                elev = 20   # 仰角
                azim = -45   # 方位角
                ax.view_init(elev=elev, azim=azim)
                fig.canvas.draw()
                action_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                action_img = action_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # 将 RGB 转换为 BGR（OpenCV 使用 BGR 格式）
                action_img = cv2.cvtColor(action_img, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    img=action_img, 
                    text=("Posture"), 
                    org=(12, 100),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=3.0, 
                    color=(0, 0, 0), 
                    thickness=10
                )

                text_img = np.ones((2400, 2400, 3), dtype=np.uint8) * 255
                
                text_img = self.create_text_image(text_img, self.language_queue, x_start=100, y_start=0, font_size=14*2.5, font_scale=2.5, text_color=(0, 0, 0), thickness=8, cogagent_label=cogagent_label)

                # Put the text on the image
                cv2.putText(
                    text_img, 
                    text=("Language"), 
                    org=(12, 100),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=3.0, 
                    color=(0, 0, 0), 
                    thickness=10
                )
                video_recorder.write(result_img)
                action_video_recorder.write(action_img)
                text_video_recorder.write(text_img)
                # cv2.imwrite('/share/home/tj16023/jinyiyang/workspace/codebase/data_helper/robot_data/data_processor/test.png', action_img)
                # cv2.imwrite("/share/home/tj16023/jinyiyang/workspace/codebase/data_helper/robot_data/data_processor/test2.png",result_img)
                # cv2.imwrite("/share/home/tj16023/jinyiyang/workspace/codebase/data_helper/robot_data/data_processor/test3.png",text_img)
                pbar.update(1)
            video_recorder.release()
            action_video_recorder.release()
            text_video_recorder.release()
            self.logger.info(f"Success visualized in - {vis_path}")
            return meta_info
    
    def search_nested_dict(self, dictionary, search_value):
        found_keys = []
        
        def recursive_search(current_dict):
            if isinstance(current_dict, dict):
                for key, value in current_dict.items():
                    if value == search_value:
                        found_keys.append(key)
                    elif isinstance(value, (dict, list)):
                        recursive_search(value)
            elif isinstance(current_dict, list):
                for item in current_dict:
                    if isinstance(item, (dict, list)):
                        recursive_search(item)
        
        recursive_search(dictionary)
        return found_keys

    def create_text_image(self, text_img, languages, x_start, y_start, font_size, font_scale, text_color, thickness, cogagent_label):
        offset = 700
        # Get the size of the text
        # text_size = cv2.getTextSize(language[-1], font, font_scale, thickness)[0]

        # # Calculate the position to center the text
        # text_x = (img.shape[1] - text_size[0]) // 2
        # text_y = (img.shape[0] + text_size[1]) // 2
        for i, language in enumerate(languages):
            max_width = 2400 * 0.8  # Use 80% of the image width
            wrapped_lines = textwrap.wrap(language, width=int(max_width / (font_size * 1.15)))
            # print(wrapped_lines)

            # Calculate total text height
            line_height = font_size * 1.5
            total_height = len(wrapped_lines) * line_height

            # Start y position
            start_y = (2400 + total_height) / 2

            for line in wrapped_lines:
                words = line.split()
                x = x_start  # Center align
                y = start_y / 5 + offset*i

                for word in words:
                    # Get word width
                    
                    word_tmp = word.replace(".", "")
                    word_tmp = word_tmp.replace(",", "")
                    label = self.search_nested_dict(cogagent_label, word_tmp)
                    if len(label) != 0:
                        color = self.color_map[label[0]]
                    else:
                        color = text_color

                    # Draw the word
                    cv2.putText(
                        text_img, 
                        text=word, 
                        org=(int(x), int(y)),
                        fontFace=cv2.FONT_ITALIC, 
                        fontScale=font_scale, 
                        color=color, 
                        thickness=thickness
                    )

                    x += len(word)*20*2.5 + 15*2.5

                start_y += line_height * 10

        return text_img
    
    def draw_action(self, ax, action, weights=None, cmap=None, color=None):
        positions = np.array([np.array(seq[:3]) for seq in action])
        rotations = np.array([np.array(seq[3:6]) for seq in action])
        gripper_widths = np.array([np.array(seq[6]) for seq in action])
        positions[:, [0, 2]] = positions[:, [2, 0]]
        # if color == None:
        #     print("gt", action)
        # else:
        #     print("pr", action)

        def plot_gripper(ax, pos, rot, width, color, finger_length=0.1, connector_length=0.05):
            # local_gripper_points = np.array([
            #     [finger_length/2, 0, width / 2],
            #     [-finger_length/2, 0, width / 2],
            #     [-finger_length/2, 0, -width / 2],
            #     [finger_length/2, 0, -width / 2],
            #     [finger_length/2 + connector_length, 0, 0] 
            # ])

            local_gripper_points = np.array([
                [finger_length/2, width / 2, 0],
                [-finger_length/2, width / 2, 0],
                [-finger_length/2, -width / 2, 0],
                [finger_length/2, -width / 2, 0],
                [finger_length/2 + connector_length, 0, 0] 
            ])

            rotation_matrix = R.from_euler('xyz', rot, degrees=False).as_matrix()

            global_gripper_points = np.dot(local_gripper_points, rotation_matrix.T) + pos
            global_gripper_points = local_gripper_points + pos
            
            ax.plot([global_gripper_points[0, 2], global_gripper_points[1, 2]],
                    [global_gripper_points[0, 1], global_gripper_points[1, 1]],
                    [global_gripper_points[0, 0], global_gripper_points[1, 0]], color=color)

            ax.plot([global_gripper_points[2, 2], global_gripper_points[3, 2]],
                    [global_gripper_points[2, 1], global_gripper_points[3, 1]],
                    [global_gripper_points[2, 0], global_gripper_points[3, 0]], color=color)

            ax.plot([global_gripper_points[3, 2], global_gripper_points[0, 2]],
                    [global_gripper_points[3, 1], global_gripper_points[0, 1]],
                    [global_gripper_points[3, 0], global_gripper_points[0, 0]], color=color)
            
            ax.plot([global_gripper_points[4, 2], (global_gripper_points[0, 2] + global_gripper_points[3, 2]) / 2],
                    [global_gripper_points[4, 1], (global_gripper_points[0, 1] + global_gripper_points[3, 1]) / 2],
                    [global_gripper_points[4, 0], (global_gripper_points[0, 0] + global_gripper_points[3, 0]) / 2], color=color)

        for i, (pos, rot, width) in enumerate(zip(positions, rotations, gripper_widths)):
            if cmap != None:
                color = cmap(weights[i]) 
            else:
                color = color 
            width = 1.0 - width/255.0
            plot_gripper(ax, pos, rot, width/20.0, finger_length=0.04, connector_length=0.04, color=color)

            if i > 0:
                prev_pos = positions[i - 1]
                ax.plot([prev_pos[2], pos[2]], [prev_pos[1], pos[1]], [prev_pos[0], pos[0]], 'o', color=color, markersize=3)

        ax.set_xlim([0.35, 0.5])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([0.2, 0.6])

        ax.set_xlabel('X Position', fontdict={'family': 'Times New Roman', 'size': 16}, labelpad=10)
        ax.set_ylabel('Y Position', fontdict={'family': 'Times New Roman', 'size': 16}, labelpad=10)
        ax.set_zlabel('Z Position', fontdict={'family': 'Times New Roman', 'size': 16}, labelpad=16)
        ax.set_title('3D Visualization of Gripper Position and Rotation')

        ax.tick_params(axis="x", labelsize=14, pad=4)
        ax.tick_params(axis="y", labelsize=14, pad=4)
        ax.tick_params(axis="z", labelsize=14, pad=10)

        labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.tight_layout()
        return ax
        
    def process(self, meta, task_infos):
        results = []
        json_paths = []
        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in filenames:
                if filename.endswith("resultV2.json"):
                    json_paths.append(os.path.join(dirpath, filename))
        if self.pool > 1:
            args_list = []
            meta_info = []
            for json_path in json_paths:  #依次读取视频文件
                args_list.append((meta_info, json_path, self.fps))
            results = self.multiprocess_run(self.vis_per_meta, args_list)
        else:
            meta_info = []
            for idx, json_path in enumerate(json_paths):  #依次读取视频文件
                self.logger.info(
                    f"Start process {idx+1}/{len(json_paths)}")
                results.append(
                    self.vis_per_meta(meta_info, json_path, self.fps))
        # for meta_infos in results:
        #     # TODO 试试看不写这句行不行
        #     meta.append(meta_infos)
        return meta, task_infos
