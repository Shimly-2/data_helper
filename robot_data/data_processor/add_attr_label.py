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
from collections import defaultdict
from robot_data.data_processor.dataset_info import label_info, cogagent_label_idx, grasp_label_idx, action_stage_idx, options, word_prompt_template, prompt_templates

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

def dictkey2list(data):
    result = []
    if isinstance(data, (list, np.ndarray, tuple)):
        for item in data:
            result.extend(dict2list(item))
    elif isinstance(data, dict):
        for value in data.keys():
            result.extend(dict2list(value))
    else:
        result.append(data)
    return result

@DATA_PROCESSER_REGISTRY.register("AddAttrLabel")
class AddAttrLabel(BaseDataProcessor):
    """get all the label and turn to one-hot"""
    def __init__(
        self,
        workspace,
        dataset_root,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.timestamp_maker = RobotTimestampsIncoder()

    def gen_per_meta(self, meta_info, json_path):
        with open(os.path.join(self.dataset_root, json_path), "r") as f:
            data = json.load(f)
        print(f"PID[{os.getpid()}: Load json file - {json_path}")

        object_name = json_path.split("/")[-3]
        epoch = json_path.split("/")[-2]
        epoch_idx = int(epoch.split("_")[-1])

        steps = data["case_info"]["steps"]
        gripper_cmd = data["frame_info"][str(steps-1)]["commended"]["gripper_closedness_commanded"]
        thr_0_9 = int(gripper_cmd / 10 * 3)
        thr_10_19 = int(gripper_cmd / 10 * 1)

        for idx in tqdm(data["frame_info"], colour='green', desc=f'PID[{os.getpid()}]'):
            action_stage = data["frame_info"][idx]["action_stage"]
            current_gripper_cmd = data["frame_info"][idx]["commended"]["gripper_closedness_commanded"]
            cur_attr_label = label_info[object_name]["attr_label"]
            attr_label = copy.deepcopy(cur_attr_label)
            cur_instruction = label_info[object_name]["GPT"]

            if action_stage == "move":
                data["frame_info"][idx]["grasp_label"] = "nocontacting"
                attr_label["position"] = cur_attr_label["position"][0]
                attr_label["deformation"] = cur_attr_label["deformation"][0]
            if action_stage == "grasp" and current_gripper_cmd < gripper_cmd - thr_0_9 and epoch_idx <= 9:
                data["frame_info"][idx]["grasp_label"] = "nocontacting"
                attr_label["position"] = cur_attr_label["position"][1]
                if len(cur_attr_label["deformation"]) == 1:
                    attr_label["deformation"] = cur_attr_label["deformation"][0]
                if len(cur_attr_label["deformation"]) == 2:
                    attr_label["deformation"] = cur_attr_label["deformation"][0]
                if len(cur_attr_label["deformation"]) == 3:
                    attr_label["deformation"] = cur_attr_label["deformation"][0]
            if action_stage == "grasp" and current_gripper_cmd >= gripper_cmd - thr_0_9 and epoch_idx <= 9:
                data["frame_info"][idx]["grasp_label"] = "grasping"
                attr_label["position"] = cur_attr_label["position"][2]
                if len(cur_attr_label["deformation"]) == 1:
                    attr_label["deformation"] = cur_attr_label["deformation"][0]
                if len(cur_attr_label["deformation"]) == 2:
                    attr_label["deformation"] = cur_attr_label["deformation"][1]
                if len(cur_attr_label["deformation"]) == 3:
                    attr_label["deformation"] = cur_attr_label["deformation"][1]
            if action_stage == "grasp" and current_gripper_cmd < gripper_cmd + thr_10_19 and epoch_idx > 9:
                data["frame_info"][idx]["grasp_label"] = "nocontacting"
                attr_label["position"] = cur_attr_label["position"][1]
                if len(cur_attr_label["deformation"]) == 1:
                    attr_label["deformation"] = cur_attr_label["deformation"][0]
                if len(cur_attr_label["deformation"]) == 2:
                    attr_label["deformation"] = cur_attr_label["deformation"][0]
                if len(cur_attr_label["deformation"]) == 3:
                    attr_label["deformation"] = cur_attr_label["deformation"][0]
            if action_stage == "grasp" and current_gripper_cmd >= gripper_cmd + thr_10_19 and epoch_idx > 9:
                data["frame_info"][idx]["grasp_label"] = "grasping"
                attr_label["position"] = cur_attr_label["position"][2]
                if len(cur_attr_label["deformation"]) == 1:
                    attr_label["deformation"] = cur_attr_label["deformation"][0]
                if len(cur_attr_label["deformation"]) == 2:
                    attr_label["deformation"] = cur_attr_label["deformation"][1]
                if len(cur_attr_label["deformation"]) == 3:
                    attr_label["deformation"] = cur_attr_label["deformation"][1]
            if action_stage == "lift":
                if not isinstance(cur_attr_label["color"], list):
                    if epoch_idx <= 9:
                        data["frame_info"][idx]["grasp_label"] = "holding"
                        if len(cur_attr_label["deformation"]) == 1:
                            attr_label["deformation"] = cur_attr_label["deformation"][0]
                        if len(cur_attr_label["deformation"]) == 2:
                            attr_label["deformation"] = cur_attr_label["deformation"][1]
                        if len(cur_attr_label["deformation"]) == 3:
                            attr_label["deformation"] = cur_attr_label["deformation"][2]
                        attr_label["position"] = cur_attr_label["position"][2]
                    else:
                        data["frame_info"][idx]["grasp_label"] = "sliped"
                        attr_label["deformation"] = cur_attr_label["deformation"][0]
                        attr_label["position"] = cur_attr_label["position"][0]
                else:
                    data["frame_info"][idx]["grasp_label"] = "holding"
                    if len(cur_attr_label["deformation"]) == 1:
                        attr_label["deformation"] = cur_attr_label["deformation"][0]
                    if len(cur_attr_label["deformation"]) == 2:
                        attr_label["deformation"] = cur_attr_label["deformation"][1]
                    if len(cur_attr_label["deformation"]) == 3:
                        attr_label["deformation"] = cur_attr_label["deformation"][2]
                    attr_label["position"] = cur_attr_label["position"][2]
                
            if isinstance(cur_attr_label["color"], list):
                if epoch_idx <= 9:
                    attr_label["color"] = cur_attr_label["color"][0]
                else:
                    attr_label["color"] = cur_attr_label["color"][1]
            
            data["frame_info"][idx]["cogagent_label"] = dict()
            data["frame_info"][idx]["cogagent_label"]["fronttop"] = attr_label
            data["frame_info"][idx]["instruction_gpt"] = cur_instruction

            for cam_view, img_path_ in data["frame_info"][idx]["cam_views"].items():
                if data["frame_info"][idx]["cam_views"][cam_view]["rgb_img_path"][0] == "/":
                    data["frame_info"][idx]["cam_views"][cam_view]["rgb_img_path"] = data["frame_info"][idx]["cam_views"][cam_view]["rgb_img_path"][1:]
                if data["frame_info"][idx]["cam_views"][cam_view]["depth_img_path"][0] == "/":
                    data["frame_info"][idx]["cam_views"][cam_view]["depth_img_path"] = data["frame_info"][idx]["cam_views"][cam_view]["depth_img_path"][1:]
            for tactile_view, img_path_ in data["frame_info"][idx]["tactile_views"].items():
                if data["frame_info"][idx]["tactile_views"][tactile_view]["rgb_img_path"][0] == "/":
                    data["frame_info"][idx]["tactile_views"][tactile_view]["rgb_img_path"] = data["frame_info"][idx]["tactile_views"][tactile_view]["rgb_img_path"][1:]
            if len(data["frame_info"][idx]["commended"]["ee_command_orientation"]) == 2:
                data["frame_info"][idx]["commended"]["ee_command_orientation"] = [-np.pi] + data["frame_info"][idx]["commended"]["ee_command_orientation"]
            
            data["frame_info"][idx]["grasp_label_idx"] = grasp_label_idx.index(data["frame_info"][idx]["grasp_label"])
            data["frame_info"][idx]["action_stage_idx"] = action_stage_idx.index(data["frame_info"][idx]["action_stage"])
        
            data["frame_info"][idx]["cogagent_label_idx"] = dict()
            for cam_view, subdict in data["frame_info"][idx]["cogagent_label"].items():
                data["frame_info"][idx]["cogagent_label_idx"][cam_view] = []
                for option, value in subdict.items():
                    actual_option = data["frame_info"][idx]["cogagent_label"][cam_view][option]
                    data["frame_info"][idx]["cogagent_label_idx"][cam_view].append(cogagent_label_idx[cam_view][option].index(actual_option))

            options_templet = dict()
            for option, label in options.items():
                option_tmp = data["frame_info"][idx]["cogagent_label"]["fronttop"][option]
                if isinstance(option_tmp, list):
                    option_tmp = option_tmp[0]
                options_templet[f"{option}_option"] = option_tmp
            options_templet["grasp_option"] = data["frame_info"][idx]["grasp_label"]
            options_templet["stage_option"] = data["frame_info"][idx]["action_stage"]
            random_templet = np.random.choice(prompt_templates)
            data["frame_info"][idx]["instruction_roml"] = random_templet.format(**options_templet)
            data["frame_info"][idx]["instruction_foml"] = prompt_templates[0].format(**options_templet)

            options_templet = dict()
            for option, label in options.items():
                option_tmp = data["frame_info"][idx]["cogagent_label"]["fronttop"][option]
                if isinstance(option_tmp, list):
                    option_tmp = option_tmp[0]
                options_templet[f"{option}_option"] = option_tmp
            options_templet["grasp_option"] = data["frame_info"][idx]["grasp_label"]
            options_templet["stage_option"] = data["frame_info"][idx]["action_stage"]
            random_templet = np.random.choice(word_prompt_template)
            data["frame_info"][idx]["instruction_rovl"] = random_templet.format(**options_templet)
            data["frame_info"][idx]["instruction_fovl"] = word_prompt_template[0].format(**options_templet)

            # print(data["frame_info"][idx]["cogagent_label"])
            # print(data["frame_info"][idx]["cogagent_label_idx"])
            # print(data["frame_info"][idx]["grasp_label"])
            # print(data["frame_info"][idx]["grasp_label_idx"])
            # print(data["frame_info"][idx]["action_stage"])
            # print(data["frame_info"][idx]["action_stage_idx"])

        json_path = json_path.replace("result", "resultV2")
        with open(os.path.join(self.dataset_root, json_path), "w") as f:
            json.dump(data, f) #, indent=4)
        return meta_info
        
    def process(self, meta, task_infos):
        results = []
        json_paths = []
        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in filenames:
                if filename.endswith("result.json"):
                    json_paths.append(os.path.join(dirpath, filename))
        if self.pool > 1:
            args_list = []
            meta_info = []
            for json_path in json_paths:  #依次读取视频文件
                args_list.append((meta, json_path))
            results = self.multiprocess_run(self.gen_per_meta, args_list)
        else:
            meta_info = []
            for idx, json_path in enumerate(json_paths):  #依次读取视频文件
                self.logger.info(
                    f"Start process {idx+1}/{len(json_paths)}")
                results.append(
                    self.gen_per_meta(meta, json_path))
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            meta.append(meta_infos)

        return meta, task_infos
