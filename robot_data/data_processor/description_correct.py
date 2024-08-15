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
import json
from robot_data.utils.utils import get_dirpath_from_key, dict2list

prompt_templet = "At 'position', there is a 'color' 'shape' 'object' made of 'material' with a 'texture' 'surface', its size is 'size', weight is 'weight', and it shows 'deformation' when interacted with."

# description = {
#     "Move":{
#         "rubikscube": "<Description> At the center of the table, there is a colorful cubic object called rubikscube made of plastic with a smooth surface, its size is medium, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of rubikscube.",

#         "tomato": "<Description> At the center of the table, there is a red and white cylindrical object called tomato soup can made of metal with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of tomato soup can.",

#         "soap": "<Description> At the center of the table, there is a blue rectangular object called soap made of soap with a smooth surface, its size is small, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of soap.", 

#         "plastic_apple": "<Description> At the center of the table, there is a red spherical object called plastic apple made of plastic with a smooth surface, its size is medium, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of plastic apple.", 

#         "two_color_hammer": "<Description> At the center of the table, there is a white and black cylindrical object called hammer made of metal with a textured surface, its size is medium, weight is heavy, and it may have no deformation when interacted with. <Instruction> Move to the place of hammer.", 

#         "sugar_box": "<Description> At the center of the table, there is a yellow and white rectangular object called sugar box made of cardboard with a smooth surface, its size is medium, weight is light, and it may have medium deformation when interacted with. <Instruction> Move to the place of sugar box.", 

#         "shampoo": "<Description> At the center of the table, there is a white and blue cylindrical object called shampoo made of plastic with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of shampoo.", 

#         "magic_clean": "<Description> At the center of the table, there is a green rectangular object called magic clean made of plastic with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of magic clean.",
#     },
#     "Grasp":{
#         "rubikscube": "<Description> The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of rubikscube.",

#         "tomato": "<Description> The tomato soup can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of tomato soup can.",

#         "soap": "<Description> The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of soap.", 

#         "plastic_apple": "<Description> The plastic apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of plastic apple.", 

#         "two_color_hammer": "<Description> The hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation when been grasped. <Instruction> Grasp the object of hammer.", 

#         "sugar_box": "<Description> The sugar box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation when been grasped. <Instruction> Grasp the object of sugar box.", 

#         "shampoo": "<Description> The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of shampoo.", 

#         "magic_clean": "<Description> The magic clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of magic clean.",
#     },
#     "Bring":{
#         "rubikscube": "<Description> The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need medium force to grasp when been lifted. <Instruction> Lift the object of rubikscube.",

#         "tomato": "<Description> The tomato soup can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of tomato soup can.",

#         "soap": "<Description> The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation and need medium force to grasp when been lifted. <Instruction> Lift the object of soap.", 

#         "plastic_apple": "<Description> The plastic apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need large force to grasp when been lifted. <Instruction> Lift the object of plastic apple.", 

#         "two_color_hammer": "<Description> The hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation and need large force to grasp when been lifted. <Instruction> Lift the object of hammer.", 

#         "sugar_box": "<Description> The sugar box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation and need light force to grasp when been lifted. <Instruction> Lift the object of sugar box.", 

#         "shampoo": "<Description> The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of shampoo.", 

#         "magic_clean": "<Description> The magic clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of magic clean.",
#     }
# }

description = {
    "Move":{
        "RubiksCube": "<Description> At the center of the table, there is a colorful cubic object called rubikscube made of plastic with a smooth surface, its size is medium, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of rubikscube.",

        "TomatoSoupCan": "<Description> At the center of the table, there is a red and white cylindrical object called tomato soup can made of metal with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of tomato soup can.",

        "soap": "<Description> At the center of the table, there is a blue rectangular object called soap made of soap with a smooth surface, its size is small, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of soap.", 

        "plastic_apple": "<Description> At the center of the table, there is a red spherical object called plastic apple made of plastic with a smooth surface, its size is medium, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of plastic apple.", 

        "two_color_hammer": "<Description> At the center of the table, there is a white and black cylindrical object called hammer made of metal with a textured surface, its size is medium, weight is heavy, and it may have no deformation when interacted with. <Instruction> Move to the place of hammer.", 

        "sugar_box": "<Description> At the center of the table, there is a yellow and white rectangular object called sugar box made of cardboard with a smooth surface, its size is medium, weight is light, and it may have medium deformation when interacted with. <Instruction> Move to the place of sugar box.", 

        "shampoo": "<Description> At the center of the table, there is a white and blue cylindrical object called shampoo made of plastic with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of shampoo.", 

        "magic_clean": "<Description> At the center of the table, there is a green rectangular object called magic clean made of plastic with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of magic clean.",
    },
    "Grasp":{
        "RubiksCube": "<Description> The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of rubikscube.",

        "TomatoSoupCan": "<Description> The tomato soup can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of tomato soup can.",

        "soap": "<Description> The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of soap.", 

        "plastic_apple": "<Description> The plastic apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of plastic apple.", 

        "two_color_hammer": "<Description> The hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation when been grasped. <Instruction> Grasp the object of hammer.", 

        "sugar_box": "<Description> The sugar box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation when been grasped. <Instruction> Grasp the object of sugar box.", 

        "shampoo": "<Description> The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of shampoo.", 

        "magic_clean": "<Description> The magic clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of magic clean.",
    },
    "Bring":{
        "RubiksCube": "<Description> The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need medium force to grasp when been lifted. <Instruction> Lift the object of rubikscube.",

        "TomatoSoupCan": "<Description> The tomato soup can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of tomato soup can.",

        "soap": "<Description> The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation and need medium force to grasp when been lifted. <Instruction> Lift the object of soap.", 

        "plastic_apple": "<Description> The plastic apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need large force to grasp when been lifted. <Instruction> Lift the object of plastic apple.", 

        "two_color_hammer": "<Description> The hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation and need large force to grasp when been lifted. <Instruction> Lift the object of hammer.", 

        "sugar_box": "<Description> The sugar box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation and need light force to grasp when been lifted. <Instruction> Lift the object of sugar box.", 

        "shampoo": "<Description> The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of shampoo.", 

        "magic_clean": "<Description> The magic clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of magic clean.",
    }
}

description = {
    "Move":{
        "RubiksCube": "The robot arm is not contact the rubikscube.",
        "plastic_apple": "The robot arm is not contact the plastic_apple.",
        "TomatoSoupCan": "The robot arm is not contact the tomato soup can.",
        "soap": "The robot arm is not contact the soap.", 
        "two_color_hammer": "The robot arm is not contact the hammer.", 
        "sugar_box": "The robot arm is not contact the sugar box.", 
        "shampoo": "The robot arm is not contact the shampoo.", 
        "magic_clean": "The robot arm is not contact the magic clean.",
    },
    "Grasp":{
        "RubiksCube": "The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. The robot arm is grasping the rubikscube.",
        "plastic_apple": "The plastic apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. The robot arm is grasping the plastic apple.",
        "TomatoSoupCan": "The tomato soup can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. The robot arm is grasping the tomato soup can.",
        "soap": "The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation when been grasped. The robot arm is grasping the soap.", 
        "two_color_hammer": "The hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation when been grasped. The robot arm is grasping the hammer.", 
        "sugar_box": "The sugar box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation when been grasped. The robot arm is grasping the sugar box.", 
        "shampoo": "The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. The robot arm is grasping the shampoo.", 
        "magic_clean": "The magic clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. The robot arm is grasping the magic clean.",
    },
    "Bring":{
        "RubiksCube": "The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need medium force to grasp when been lifted. The robot arm is lifting the rubikscube.",
        "plastic_apple": "The plastic apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need large force to grasp when been lifted. The robot arm is lifting the plastic apple.", 
        "TomatoSoupCan": "The tomato soup can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. The robot arm is lifting the tomato soup can.",
        "soap": "The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation and need medium force to grasp when been lifted. The robot arm is lifting the soap.",  
        "two_color_hammer": "The hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation and need large force to grasp when been lifted. The robot arm is lifting the hammer.", 
        "sugar_box": "The sugar box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation and need light force to grasp when been lifted. The robot arm is lifting the sugar box.", 
        "shampoo": "The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. The robot arm is lifting the shampoo.", 
        "magic_clean": "The magic clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. The robot arm is lifting the magic clean.",
    }
}

@DATA_PROCESSER_REGISTRY.register("DescriptionCorrect")
class DescriptionCorrect(BaseDataProcessor):
    """decription correct"""
    def __init__(
        self,
        workspace,
        dataset_root,
        suffix=None,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.dataset_root = dataset_root
        self.timestamp_maker = RobotTimestampsIncoder()
        self.suffix = suffix

    def split_per_meta(self, meta_info, json_path):
        with open(json_path,"r") as f:
            meta_info = json.load(f)
        print(f"PID[{os.getpid()}: Load json file - {json_path}")
        root_path = get_dirpath_from_key(json_path, "raw_meta")
        frame_infos = meta_info["frame_info"]
        window_set = []
        for idx in tqdm(meta_info["frame_info"], colour='green', desc=f'PID[{os.getpid()}]'):
            if "Move" in meta_info["frame_info"][idx]["instruction"]:
                object_name = meta_info["frame_info"][idx]["instruction"].split(" ")[3].split(".")[0]
                try:
                    if self.suffix != None:
                        meta_info["frame_info"][idx][f"instruction_{self.suffix}"] = description["Move"][object_name]
                    else:
                        meta_info["frame_info"][idx]["instruction"] = description["Move"][object_name]
                except:
                    print("Missing key:", object_name)
            elif "Grasp" in meta_info["frame_info"][idx]["instruction"]:
                object_name = meta_info["frame_info"][idx]["instruction"].split(" ")[2].split(".")[0]
                try:
                    if self.suffix != None:
                        meta_info["frame_info"][idx][f"instruction_{self.suffix}"] = description["Grasp"][object_name]
                    else:
                        meta_info["frame_info"][idx]["instruction"] = description["Grasp"][object_name]
                except:
                    print("Missing key:", object_name)
            elif "Bring" in meta_info["frame_info"][idx]["instruction"]:
                object_name = meta_info["frame_info"][idx]["instruction"].split(" ")[3].split(".")[0]
                try:
                    if self.suffix != None:
                        meta_info["frame_info"][idx][f"instruction_{self.suffix}"] = description["Bring"][object_name]
                    else:
                        meta_info["frame_info"][idx]["instruction"] = description["Bring"][object_name]
                except:
                    print("Missing key:", object_name)
        with open(json_path, "w") as f:
            json.dump(meta_info, f) #, indent=4)
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
                args_list.append((meta_info, json_path))
            results = self.multiprocess_run(self.split_per_meta, args_list)
        else:
            meta_info = []
            for idx, json_path in enumerate(json_paths):  #依次读取视频文件
                filename = osp.split(json_path)[-1]
                # save_new_filename = osp.join(self.save_root, filename)
                self.logger.info(
                    f"Start process {idx+1}/{len(json_paths)}")
                results.append(
                    self.split_per_meta(meta_info, json_path))
        for meta_infos in results:
            # TODO 试试看不写这句行不行
            meta.append(meta_infos)
        return meta, task_infos