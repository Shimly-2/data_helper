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
from robot_data.utils.utils import get_dirpath_from_key
import roboticstoolbox as rtb
import csv

def dict2list(data):
    result = []
    if isinstance(data, (list, np.ndarray, tuple)):
        for item in data:
            result.extend(dict2list(item))
    elif isinstance(data, dict):
        for value in data.values():
            result.extend(dict2list(value))
    else:
        result.append(data)
    return result

@DATA_PROCESSER_REGISTRY.register("GenPoseltMetaINFO")
class GenPoseltMetaINFO(BaseDataProcessor):
    """生产meta数据，将所有视频，图片路径变为绝对路径"""
    def __init__(
        self,
        workspace,
        dataset_root,
        collect_keys=["depth", "gelsight", "rgb", "side_cam", "top_cam"],
        collect_csvs=["f_t", "gripper", "label", "robot", "stages"],
        cam_views=["side", "top", "fronttop"], 
        cam_views_mapping={"side":"side_cam", "top":"top_cam", "fronttop":"rgb"}, # "fronttop", "side", "topdown", "root" ,
        tactile_views=["gelsightL"],
        tactile_views_mapping={"gelsightL":"gelsight"},
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.timestamp_maker = RobotTimestampsIncoder()
        self.collect_keys = collect_keys
        self.collect_csvs = collect_csvs
        self.cam_views = cam_views
        self.tactile_views = tactile_views
        self.cam_views_mapping = cam_views_mapping
        self.tactile_views_mapping = tactile_views_mapping
        self.robot = rtb.models.UR5()

    def gen_per_meta(self, meta_info, json_path):
        case_name = json_path.split("/")[-2]
        data = dict()
        data["case_info"] = dict()
        data["frame_info"] = dict()
        data["case_info"]["img_size"] = [320, 256]
        idx = case_name.split('_')[-1][4:]
        data["case_info"]["epoch"] = f"epoch_{idx}"
        data["case_info"]["task_name"] = "grasp_multi_objects"
        data["case_info"]["case_name"] = "poselt"
        object_name = case_name.split('_')[0]
        data["case_info"]["object_name"] = object_name
        data["case_info"]["fps"] = 30
        data["case_info"]["timestamp"] = case_name.split('_')[1]
        data["case_info"]["arm_dof"] = 6
        
        collect_info = dict()
        min_step = 1e10
        for collect_key in self.collect_keys:
            filename_list = [f for f in os.listdir(os.path.join(self.dataset_root, case_name, collect_key))]
            sorted_filename_list = sorted(filename_list, key=lambda x: int(x.split('_')[-2]))
            collect_info[collect_key] = sorted_filename_list
            if min_step > len(collect_info[collect_key]):
                min_step = len(collect_info[collect_key])
        
        for collect_csv in self.collect_csvs:
            with open(os.path.join(self.dataset_root, case_name, f"{collect_csv}.csv"), mode='r', newline='') as file:
                csv_reader = csv.reader(file)
                if collect_csv == "label" or collect_csv == "stages":
                    pass
                else:
                    next(csv_reader)
                collect_info[collect_csv] = []
                for row in csv_reader:
                    if collect_csv == "label" or collect_csv == "stages":
                        numeric_row = [item for item in row]
                    else:
                        numeric_row = [float(item) for item in row]
                    collect_info[collect_csv].append(numeric_row)
        current_stage = 0
        
        for collect_key in self.collect_keys:
            collect_info[f"{collect_key}_fps"] = round(len(collect_info[collect_key]) / min_step) - 1
            # print(collect_key, collect_info[f"{collect_key}_fps"], len(collect_info[collect_key]) / min_step)
        for collect_csv in self.collect_csvs:
            collect_info[f"{collect_csv}_fps"] = round(len(collect_info[collect_csv]) / min_step) - 1
            # print(collect_csv, collect_info[f"{collect_csv}_fps"], len(collect_info[collect_csv]) / min_step)

        pbar_i = tqdm(total=min_step-1, colour="green", desc=f"Epoch-{idx} collecting {object_name}")
        # Throw the last
        for j in range(min_step - 1):
            data["frame_info"][j] = dict()
            data["frame_info"][j]["idx"] = j
            data["frame_info"][j]["cam_views"] = dict()
            for cam_view in self.cam_views:
                fps = collect_info[f"{self.cam_views_mapping[cam_view]}_fps"]
                offset = j + j * fps
                if offset > len(collect_info[self.cam_views_mapping[cam_view]]) - 1:
                    offset = len(collect_info[self.cam_views_mapping[cam_view]]) - 1
                image_name = collect_info[self.cam_views_mapping[cam_view]][offset]
                # print(cam_view, j, fps, j + j * fps, len(collect_info[self.cam_views_mapping[cam_view]]), image_name)
                data["frame_info"][j]["cam_views"][cam_view] = dict()
                # data["frame_info"][j]["cam_views"][cam_view]["rgb_video_path"] = os.path.join("raw_meta", object_name, f"epoch_{i}", f"{self.task_env.cam_views[k]}.mp4")
                data["frame_info"][j]["cam_views"][cam_view]["rgb_img_path"] = os.path.join(json_path.split("/")[-2], self.cam_views_mapping[cam_view], image_name)
            data["frame_info"][j]["tactile_views"] = dict()
            for tactile_view in self.tactile_views:
                fps = collect_info[f"{self.tactile_views_mapping[tactile_view]}_fps"]
                offset = j + j * fps
                if offset > len(collect_info[self.tactile_views_mapping[tactile_view]]) - 1:
                    offset = len(collect_info[self.tactile_views_mapping[tactile_view]]) - 1
                image_name = collect_info[self.tactile_views_mapping[tactile_view]][offset]
                data["frame_info"][j]["tactile_views"][tactile_view] = dict()
                # data["frame_info"][j]["tactile_views"][tactile_view]["rgb_video_path"] = os.path.join("raw_meta", object_name, f"epoch_{i}", f"{self.task_env.video_recorder.tactile_views[k]}.mp4")
                data["frame_info"][j]["tactile_views"][tactile_view]["rgb_img_path"] = os.path.join(json_path.split("/")[-2], self.tactile_views_mapping[tactile_view], image_name)
            
            # get force
            fps_f = collect_info[f"f_t_fps"]
            tmp_nF = collect_info["f_t"][j * fps_f: (j + 1) * fps_f]
            tmp_nF = [row[1:4] for row in tmp_nF]
            tmp_lF = collect_info["f_t"][j * fps_f: (j + 1) * fps_f]
            tmp_lF = [row[4:7] for row in tmp_lF]
            tmp_nF = np.mean(np.array(tmp_nF), axis=0)
            tmp_lF = np.mean(np.array(tmp_lF), axis=0)
            normalForce, lateralForce = list(tmp_nF), list(tmp_lF)

            data["frame_info"][j]["tar_object_states"] = dict()
            data["frame_info"][j]["commended"] = dict()
            data["frame_info"][j]["tar_object_states"]["tar_object_position"] = [0.0, 0.0, 0.0]
            data["frame_info"][j]["tar_object_states"]["tar_object_orientation"] = [0.0, 0.0, 0.0]

            jointstates = dict()
            joints = ["eibow", "shoulder_l", "shoulder_p", "wrist1", "wrist2", "wrist3"]
            fps_j = collect_info[f"robot_fps"]
            for idx, joint in enumerate(joints):
                tmp_joint = collect_info["robot"][j * fps_j: (j + 1) * fps_j]
                tmp_joint = [row[idx+1] for row in tmp_joint]
                tmp_joint = np.mean(np.array(tmp_joint))
                jointstates[joint] = tmp_joint
            data["frame_info"][j]["jointstates"] = jointstates
            
            fps_g = collect_info[f"gripper_fps"]
            tmp_gripper = collect_info["gripper"][j * fps_g: (j + 1) * fps_g]
            tmp_gripper = [row[1] for row in tmp_gripper]
            tmp_gripper = np.mean(np.array(tmp_gripper))
            gripper_closedness_commanded = tmp_gripper
            data["frame_info"][j]["commended"]["gripper_closedness_commanded"] = gripper_closedness_commanded
            data["frame_info"][j]["distance_to_target"] = 0.0
            data["frame_info"][j]["normalForceL"] = normalForce
            data["frame_info"][j]["normalForceR"] = normalForce
            data["frame_info"][j]["lateralForceL"] = list(lateralForce)
            data["frame_info"][j]["lateralForceR"] = list(lateralForce)
            if collect_info["label"][0][1] == "pass":
                data["frame_info"][j]["success"] = True
            else:
                data["frame_info"][j]["success"] = False

            # print(delta_ee_pos, delta_ee_ori)#, p.getEulerFromQuaternion(delta_ee_ori)
            
            # absolute pose
            delta_ee_pos, delta_ee_ori = self.cal_FK(self.robot, dict2list(jointstates))
            data["frame_info"][j]["commended"]["ee_command_position"] = list(delta_ee_pos) # list(self.task_env.CR5_robot.get_end_effector_pos())
            data["frame_info"][j]["commended"]["ee_command_orientation"] = list(delta_ee_ori) # list(self.task_env.CR5_robot.get_end_effector_ori())

            # label
            current_timestamp = int(collect_info["rgb"][j].split(".")[0].split("_")[-1])
            current_stage_timestamp = int(collect_info["stages"][current_stage][1])
            if current_timestamp <= current_stage_timestamp:
                data["frame_info"][j]["action_stage"] = collect_info["stages"][current_stage][0]
            else:
                current_stage += 1
                data["frame_info"][j]["action_stage"] = collect_info["stages"][current_stage][0]
            if current_stage <= 3:
                data["frame_info"][j]["grasp_label"] = collect_info["label"][current_stage][1]
            elif current_stage == 4 or current_stage == 5:
                data["frame_info"][j]["grasp_label"] = collect_info["label"][3][1]
            # print(collect_info["rgb"][j], collect_info["rgb"][j].split(".")[0].split("_")[-1], collect_info["label"][j], collect_info["stages"][j])
            # data["frame_info"][j]["label"] = labels["Move"][object_name]
            # data["frame_info"][j]["action"] = "move"
            # data["frame_info"][j]["contact_state"] = "no-contact"

            pbar_i.set_postfix({
                # 'mode': '%s' % (mode),
                # 'dis_to_target': '%.3f' % (distance_to_target),
                'gripper_cmd': '%.3f' % gripper_closedness_commanded ,
            })
            pbar_i.update(1)
        data["case_info"]["steps"] = j + 1
        with open(os.path.join(self.dataset_root, json_path), 'w') as file:
            json.dump(data, file)  #, indent=4)
    
        return meta_info
    
    def cal_FK(self, robot, joint):
        end_effector_pose = robot.fkine(np.array(joint))
        position = end_effector_pose.t
        rotation_matrix = end_effector_pose.R
        euler_angles = end_effector_pose.rpy()  # ZYX 
        return position, euler_angles
        
    def process(self, meta, task_infos):
        results = []
        json_paths = []
        json_dir_list = [name for name in os.listdir(self.dataset_root) if os.path.isdir(os.path.join(self.dataset_root, name))]
        if self.pool > 1:
            args_list = []
            meta_info = []
            for json_dir in json_dir_list:  #依次读取视频文件
                json_path = os.path.join(json_dir, "result.json")
                args_list.append((meta_info, json_path))
            results = self.multiprocess_run(self.gen_per_meta, args_list)
        else:
            meta_info = []
            for idx, json_dir in enumerate(json_dir_list):  #依次读取视频文件
                filename = osp.split(json_dir)[-1]
                json_path = os.path.join(json_dir, "result.json")
                # save_new_filename = osp.join(self.save_root, filename)
                self.logger.info(
                    f"Start process {idx+1}/{len(json_dir_list)}")
                results.append(
                    self.gen_per_meta(meta_info, json_path))
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            if meta_infos == []:
                pass
            else:
                meta.append(meta_infos)
        return meta, task_infos
