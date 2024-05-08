# from .BaseDataProcessor import BaseDataProcessor
# import cv2
# import time
# import os
# import json
# from tqdm import tqdm
# from tqdm import trange
# import math
# import multiprocessing
# import os.path as osp
# import numpy as np
# import copy
# from robot_data.utils.registry_factory import DATA_PROCESSER_REGISTRY
# from robot_data.utils.robot_timestamp import RobotTimestampsIncoder
# from robot_data.utils.utils import get_dirpath_from_key

# @DATA_PROCESSER_REGISTRY.register("TactilePerception")
# class TactilePerception(BaseDataProcessor):
#     """生产meta数据，将所有视频，图片路径变为绝对路径"""
#     def __init__(
#         self,
#         workspace,
#         dataset_root,
#         **kwargs,
#     ):
#         super().__init__(workspace, **kwargs)
#         self.dataset_root = dataset_root
#         self.timestamp_maker = RobotTimestampsIncoder()

#     def gen_per_meta(self, meta_info, json_path):
#         with open(json_path,"r") as f:
#             meta_info = json.load(f)
#         print(f"PID[{os.getpid()}: Load json file - {json_path}")
#         case_name = meta_info["case_info"]["case_name"]
#         task_name = meta_info["case_info"]["task_name"]
#         case_name_evl = json_path.split("/")[-5]
#         task_name_evl = json_path.split("/")[-6]
#         assert case_name == case_name_evl, "Not the same case: [meta]--<{}>, [config]--<{}>".format(case_name, case_name_evl)
#         assert task_name == task_name_evl, "Not the same task: [meta]--<{}>, [config]--<{}>".format(task_name, task_name_evl)
#         root_path = get_dirpath_from_key(json_path, "raw_meta")
#         frame_infos = meta_info["frame_info"]
#         for idx in frame_infos:
#             frame_info = frame_infos[idx]
#             for cam_view in frame_info["cam_views"]:
#                 view_info = frame_info["cam_views"][cam_view]
#                 view_info["rgb_video_path"] = os.path.join(root_path, view_info["rgb_video_path"])
#                 view_info["rgb_img_path"] = os.path.join(root_path, view_info["rgb_img_path"])
#             for tactile_view in frame_info["tactile_views"]:
#                 view_info = frame_info["tactile_views"][tactile_view]
#                 view_info["rgb_video_path"] = os.path.join(root_path, view_info["rgb_video_path"])
#                 view_info["rgb_img_path"] = os.path.join(root_path, view_info["rgb_img_path"])
#         return meta_info
        
#     def process(self, meta, task_infos):
#         results = []
#         json_paths = []
#         for dirpath, dirnames, filenames in os.walk(self.dataset_root):
#             for filename in filenames:
#                 if filename.endswith("result.json"):
#                     json_paths.append(os.path.join(dirpath, filename))
#         if self.pool > 1:
#             args_list = []
#             meta_info = []
#             for json_path in json_paths:  #依次读取视频文件
#                 args_list.append((meta_info, json_path))
#             results = self.multiprocess_run(self.gen_per_meta, args_list)
#         else:
#             for idx, json_path in enumerate(json_paths):  #依次读取视频文件
#                 filename = osp.split(json_path)[-1]
#                 # save_new_filename = osp.join(self.save_root, filename)
#                 self.logger.info(
#                     f"Start process {idx+1}/{len(json_paths)}")
#                 results.append(
#                     self.gen_per_meta(meta_info, json_path))
#         for meta_infos in results:
#             # TODO 试试看不写这句行不行
#             meta.append(meta_infos)
#         return meta, task_infos


vocabulary = {
    0: "object",
    1: "material",
    2: "texture",
    3: "surface",
    4: "position",
    5: "shape",
    6: "size",
    7: "weight",
    8: "color",
    9: "deformation"
}

descriptions = {
    "object": [
        "ball", "cup", "book", "chair", "pen", "bottle", "box", "bag", "lamp", "vase",
        "mirror", "table", "spoon", "fork", "plate", "bowl", "pillow", "sofa", "toy", "umbrella",
        "keyboard", "mouse", "headphones", "speaker", "monitor", "laptop", "phone", "tablet", "remote", "camera",
        "clock", "picture", "poster", "sculpture", "painting", "carpet", "curtain", "blinds", "rug", "mat"
    ],
    "material": [
        "wood", "metal", "plastic", "glass", "ceramic", "leather", "cotton", "silk", "wool", "linen",
        "paper", "stone", "rubber", "concrete", "steel", "aluminum", "copper", "iron", "gold", "silver",
        "bronze", "brass", "titanium", "platinum", "chrome", "nickel", "tin", "lead", "zinc", "graphite",
        "marble", "granite", "slate", "sandstone", "limestone", "quartz", "jade", "obsidian", "amber", "pearl"
    ],
    "texture": [
        "smooth", "rough", "bumpy", "coarse", "grainy", "fuzzy", "slippery", "sticky", "silky", "velvety",
        "bristly", "fluffy", "hairy", "prickly", "gritty", "powdery", "scaly", "spongy", "waxy", "leathery",
        "nubby", "ribbed", "grooved", "ridged", "knurled", "striated", "woven", "knitted", "braided", "twisted",
        "crimped", "corded", "looped", "knotted", "matted", "tufted", "felted", "fleecy", "downy", "plush"
    ],
    "surface": [
        "polished", "matte", "glossy", "satin", "brushed", "patterned", "embossed", "engraved", "etched", "carved",
        "molded", "textured", "ribbed", "corrugated", "perforated", "dimpled", "wrinkled", "cracked", "peeled", "worn",
        "scratched", "chipped", "dented", "tarnished", "rusted", "weathered", "faded", "stained", "discolored", "marbled",
        "speckled", "freckled", "stippled", "mottled", "veined", "grained", "streaked", "flecked", "spattered", "blotched"
    ],
    "position": [
        "top", "bottom", "left", "right", "center", "corner", "edge", "side", "front", "back",
        "inside", "outside", "above", "below", "behind", "beside", "between", "under", "over", "across",
        "northward", "southward", "eastward", "westward", "upward", "downward", "inward", "outward", "forward", "backward",
        "clockwise", "counterclockwise", "horizontal", "vertical", "diagonal", "parallel", "perpendicular", "adjacent", "proximate", "remote"
    ],
    "shape": [
        "square", "rectangular", "circular", "oval", "triangular", "spherical", "cylindrical", "conical", "pyramidal", "cuboid",
        "prismatic", "polyhedral", "ellipsoidal", "amorphous", "irregular", "symmetric", "asymmetric", "concave", "convex", "spiral",
        "helical", "zig-zag", "wavy", "curvy", "undulating", "sinuous", "serpentine", "twisted", "coiled", "interlaced",
        "crisscrossed", "latticed", "meshed", "reticulated", "honeycombed", "ribbed", "corrugated", "grooved", "fluted", "scalloped"
    ],
    "size": [
        "tiny", "small", "medium", "large", "huge", "miniature", "compact", "colossal", "gigantic", "mammoth",
        "microscopic", "diminutive", "petite", "modest", "substantial", "immense", "vast", "towering", "expansive", "gargantuan",
        "minuscule", "infinitesimal", "minute", "teeny", "wee", "puny", "bantam", "pint-sized", "bite-sized", "dainty",
        "ample", "capacious", "voluminous", "cavernous", "monumental", "prodigious", "stupendous", "titanic", "leviathan", "behemoth"
    ],
    "weight": [
        "feather-light", "lightweight", "middleweight", "heavyweight", "hefty", "weighty", "bulky", "cumbersome", "ponderous", "burdensome",
        "massive", "substantial", "negligible", "trivial", "significant", "considerable", "overwhelming", "crushing", "oppressive", "backbreaking",
        "gossamer", "wispy", "airy", "gauzy", "diaphanous", "flimsy", "insubstantial", "imponderable", "ethereal", "delicate",
        "robust", "sturdy", "solid", "stout", "brawny", "strapping", "muscular", "beefy", "burly", "Herculean"
    ],
    "color": [
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
        "gray", "beige", "tan", "maroon", "crimson", "scarlet", "indigo", "turquoise", "teal", "magenta",
        "vermilion", "burgundy", "plum", "lavender", "mauve", "fuchsia", "puce", "chartreuse", "olive", "aquamarine",
        "cerulean", "sapphire", "periwinkle", "lilac", "orchid", "fawn", "taupe", "sepia", "ecru", "ivory"
    ],
    "deformation": [
        "bend", "twist", "crumple", "wrinkle", "crease", "fold", "dent", "buckle", "warp", "contort",
        "stretch", "compress", "squash", "flatten", "bulge", "swell", "bloat", "shrink", "shrivel", "pucker",
        "crinkle", "curl", "crimp", "rumple", "scrunch", "corrugate", "pleat", "furrow", "dimple", "ripple",
        "undulate", "wave", "flute", "ruffle", "frill", "flounce", "pleat", "gather", "tuck", "drape"
    ]
}


templet = [
    # 新增的20条模板如下:
    "At {position}, there is a {color} {shape} {object} made of {material} with a {texture} {surface}, its size is {size}, weight is {weight}, and it shows {deformation} when interacted with.",
"Observe a {material} {object} of {shape} form, {color} in color, with a {texture} {surface} exhibiting {deformation} upon interaction, located at {position}, having a {size} size and {weight} weight.",
"A {weight} {color} {shape} {size} {object} crafted from {material} with a {texture} {surface} that {deformation} when touched, can be found at {position}.",
"This {material} {texture} {surface} {object} at {position} is {color} in color, {shape} in shape, {size} in size, {weight} in weight, and displays {deformation} when interacted with.",
"I can see a {shape} {object} made of {material}, with a {texture} {surface} that {deformation} upon interaction, located at {position}, having a {color} color, {size} size, and {weight} weight.",
"Placed at {position} is a {color} {material} {object} of {shape} form, with a {texture} {surface} that exhibits {deformation} when touched, its size is {size} and weight is {weight}.",
"The {material} {object} I'm examining has a {texture} {surface} that {deformation} upon interaction, it's {color} in color, {shape} in shape, {size} in size, {weight} in weight, and located at {position}.",
"At {position}, you'll find a {shape} {size} {color} {object} made of {material}, with a {texture} {surface} that {deformation} when touched, weighing {weight}.",
"Situated at {position} is a {weight} {color} {shape} {material} {object} with a {texture} {surface}, exhibiting {deformation} when interacted with, and having a {size} size.",
"Observe this {shape} {size} {material} {object} at {position}, with a {texture} {surface} that {deformation} upon interaction, it's {color} in color and weighs {weight}.",
"Behold a {color} {shape} {material} {object} with a {texture} {surface} at {position}, weighing {weight} and measuring {size}, which {deformation} when interacted with.",
"Located at {position} is a {shape} {object} made of {material}, exhibiting {deformation} on its {texture} {surface} when touched, colored {color}, sized {size}, and weighing {weight}.",
"A {material} {object} of {shape} form, {color} in color, can be found at {position}, with a {texture} {surface} that {deformation} upon interaction, measuring {size} in size and {weight} in weight.",
"Here we have a {weight} {color} {shape} {size} {object} crafted from {material}, with a {texture} {surface} that {deformation} when touched, situated at {position}.",
"Observe this {texture} {surface} {deformation} {color} {shape} {size} {weight} {material} {object} located at {position}.",
"At {position}, you'll find a {color} {material} {object} of {shape} form, with a {texture} {surface} that {deformation} when interacted with, measuring {size} in size and {weight} in weight.",
"Inspect a {shape} {size} {material} {object} at {position}, with a {texture} {surface} that {deformation} upon interaction, colored {color} and weighing {weight}.",
"This {weight} {color} {shape} {object} made of {material}, with a {texture} {surface} that {deformation} when touched, can be found at {position}, measuring {size} in size.",
"Located at {position} is a {material} {object} of {shape} form, {color} in color, with a {texture} {surface} that {deformation} upon interaction, weighing {weight} and sized {size}.",
"Observe a {size} {weight} {color} {shape} {object} crafted from {material}, with a {texture} {surface} that {deformation} when interacted with, situated at {position}.",
"Behold this {shape} {material} {object} at {position}, with a {texture} {surface} that {deformation} upon interaction, colored {color}, measuring {size} in size and {weight} in weight.",
"At {position}, you'll find a {weight} {material} {texture} {surface} {object} of {shape} form, {color} in color, exhibiting {deformation} when touched, and measuring {size} in size.",
"Inspect a {color} {shape} {size} {object} made of {material}, with a {texture} {surface} that {deformation} upon interaction, located at {position} and weighing {weight}.",
"This {texture} {surface} {deformation} {material} {object} at {position} is {color} in color, {shape} in shape, {size} in size, and {weight} in weight.",
"Observe a {weight} {color} {material} {object} of {shape} form, with a {texture} {surface} that {deformation} when touched, located at {position} and measuring {size} in size.",
"Situated at {position} is a {shape} {size} {texture} {surface} {object} crafted from {material}, {color} in color, exhibiting {deformation} when interacted with, and weighing {weight}.",
"At {position}, you'll find a {material} {object} of {shape} form, with a {texture} {surface} that {deformation} upon interaction, colored {color}, sized {size}, and weighing {weight}.",
"Behold a {weight} {size} {color} {shape} {material} {object} with a {texture} {surface} that {deformation} when touched, located at {position}.",
"Inspect this {texture} {surface} {color} {shape} {size} {object} made of {material} at {position}, exhibiting {deformation} when interacted with and weighing {weight}.",
"Observe a {shape} {material} {object} of {size} dimensions, {color} in color, with a {texture} {surface} that {deformation} upon interaction, located at {position} and weighing {weight}."
]

if __name__ == "__main__":
    print(0)
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('/mnt/afs/jinyiyang/workspace/codebase/rt1-cr5/checkpoints/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        '/mnt/afs/jinyiyang/workspace/codebase/rt1-cr5/checkpoints/cogvlm-grounding-base-hf',
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_safetensors=True
    ).to('cuda').eval()

    prompt = 'Choose the most suitable material for grasped object in the picture, for each key in the dict below, choose one description.'
    query = 'Can you provide a description of the image and include the coordinates [[x0,y0,x1,y1]] for each mentioned object?'
    image = Image.open(requests.get('/mnt/afs/jinyiyang/workspace/codebase/data_helper/datasets/demo_img/grasp_bottle.png', stream=True).raw).convert('RGB')
    inputs = model.build_conversation_input_ids(tokenizer, query=query, images=[image])
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": False}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0]))