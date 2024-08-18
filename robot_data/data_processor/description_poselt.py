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

descriptions = {
    "object": [
        "rubiks_cube", "tomato_soup_can", "soap", "plastic_apple", "two_color_hammer", "sugar_box", "shampoo", "magic_clean"
    ],
    "material": [
        "plastic", "metal", "glycerin", "glass", "cardboard"
    ],
    "texture": [
        "smooth", "rough"
    ],
    "surface": [
        "flat", "glossy"
    ],
    "position": [
        "center", 
    ],
    "shape": [
        "cuboid", "cylindrical", "rectangular", "round", "oval"
    ],
    "size": [
        "small", "medium"
    ],
    "weight": [
        "light", "heavy"
    ],
    "color": [
        "colorful", "red", "blue", "silver", "white", "green"
    ],
    "deformation": [
        "no-deformation", "slight-deformation"
    ]
}

labels = {
    "Move":{
        "RubiksCube": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "TomatoSoupCan": [1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
        "soap": [2, 2, 0, 0, 0, 2, 0 ,0, 2, 0], 
        "plastic_apple": [3, 0, 0, 0, 0, 3, 0, 0, 1, 0], 
        "two_color_hammer": [4, 1, 1, 0, 0, 4, 1, 1, 3, 0], 
        "sugar_box": [5, 4, 0, 0, 0, 2, 1, 0, 4, 1], 
        "shampoo": [6, 0, 0, 1, 0, 1, 1, 0, 0, 0], 
        "magic_clean": [7, 0, 0, 0, 0, 1, 1, 1, 5, 0],
    },
    "Grasp":{
        "RubiksCube": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "TomatoSoupCan": [1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
        "soap": [2, 2, 0, 0, 0, 2, 0 ,0, 2, 0], 
        "plastic_apple": [3, 0, 0, 0, 0, 3, 0, 0, 1, 0], 
        "two_color_hammer": [4, 1, 1, 0, 0, 4, 1, 1, 3, 0], 
        "sugar_box": [5, 4, 0, 0, 0, 2, 1, 0, 4, 1], 
        "shampoo": [6, 0, 0, 1, 0, 1, 1, 0, 0, 0], 
        "magic_clean": [7, 0, 0, 0, 0, 1, 1, 1, 5, 0],
    },
    "Bring":{
        "RubiksCube": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "TomatoSoupCan": [1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
        "soap": [2, 2, 0, 0, 0, 2, 0 ,0, 2, 0], 
        "plastic_apple": [3, 0, 0, 0, 0, 3, 0, 0, 1, 0], 
        "two_color_hammer": [4, 1, 1, 0, 0, 4, 1, 1, 3, 0], 
        "sugar_box": [5, 4, 0, 0, 0, 2, 1, 0, 4, 1], 
        "shampoo": [6, 0, 0, 1, 0, 1, 1, 0, 0, 0], 
        "magic_clean": [7, 0, 0, 0, 0, 1, 1, 1, 5, 0],
    }
}

prompt_templates = [
    """"At {position_option}, there is a {color_option} {shape_option} {object_option} made of {material_option} with a {texture_option} {surface_option}, its size is {size_option}, weight is {weight_option}, and it shows {deformation_option} when interacted with.""",

    """In the {position_option}, there's a {size_option} {color_option} {object_option} with a {texture_option} {surface_option}. It's made of {material_option}, shaped like a {shape_option}, weighs {weight_option}, and {deformation_option} upon interaction.""",

    """A {color_option} {object_option} is located {position_option}. It has a {texture_option} {surface_option}, is {size_option} in size, {shape_option} in shape, made from {material_option}, weighs {weight_option}, and {deformation_option} when touched.""",

    """Situated {position_option} is a {material_option} {object_option}. It's {color_option}, {size_option}, {shape_option}, with a {texture_option} {surface_option}. The object weighs {weight_option} and {deformation_option} under pressure.""",

    """The {position_option} features a {shape_option} {object_option}. It's {color_option}, {size_option}, made of {material_option}, has a {texture_option} {surface_option}, weighs {weight_option}, and {deformation_option} when force is applied.""",

    """A {size_option} {object_option} can be found {position_option}. It's {color_option}, {shape_option}, crafted from {material_option}, with a {texture_option} {surface_option}. Its weight is {weight_option} and it {deformation_option} upon interaction.""",

    """Observe a {color_option} {object_option} {position_option}. This {size_option} item is {shape_option}, made of {material_option}, has a {texture_option} {surface_option}, weighs {weight_option}, and {deformation_option} when handled.""",

    """Located {position_option} is a {material_option} {object_option}. It's {color_option}, {size_option}, {shape_option} in form, with a {texture_option} {surface_option}. The object has a weight of {weight_option} and {deformation_option} under stress.""",

    """A {texture_option} {surface_option} characterizes the {color_option} {object_option} {position_option}. It's {size_option}, {shape_option}, made from {material_option}, weighs {weight_option}, and {deformation_option} when force is exerted.""",

    """The {position_option} hosts a {shape_option} {object_option}. Its {color_option} surface is {texture_option}, it's made of {material_option}, {size_option} in dimensions, weighs {weight_option}, and {deformation_option} upon physical interaction.""",

    """Examine the {color_option} {object_option} at {position_option}. This {material_option} item is {size_option}, {shape_option}, has a {texture_option} {surface_option}, weighs {weight_option}, and {deformation_option} when pressure is applied.""",

    """A {shape_option} {object_option} is present {position_option}. It's {color_option}, {size_option}, constructed from {material_option}, features a {texture_option} {surface_option}, has a weight of {weight_option}, and {deformation_option} when manipulated.""",

    """Notice the {color_option} {object_option} {position_option}. This {shape_option} object is {size_option}, made of {material_option}, has a {texture_option} {surface_option}, weighs {weight_option}, and {deformation_option} under external forces.""",

    """Positioned {position_option} is a {size_option} {object_option}. It's {color_option}, {shape_option} in form, crafted from {material_option}, with a {texture_option} {surface_option}. Its weight is {weight_option} and it {deformation_option} when pressed.""",

    """The {position_option} contains a {material_option} {object_option}. It's {color_option}, {size_option}, {shape_option}, with a {texture_option} {surface_option}. The item weighs {weight_option} and {deformation_option} upon contact.""",

    """A {texture_option} {object_option} is visible {position_option}. It's {color_option}, {size_option}, {shape_option}, made of {material_option}, has a {surface_option} surface, weighs {weight_option}, and {deformation_option} when force is applied.""",

    """Discover a {color_option} {object_option} {position_option}. This {material_option} item is {size_option}, {shape_option}, has a {texture_option} {surface_option}, weighs {weight_option}, and {deformation_option} under physical stress.""",

    """The {position_option} area includes a {shape_option} {object_option}. It's {color_option}, {size_option}, made from {material_option}, has a {texture_option} {surface_option}, weighs {weight_option}, and {deformation_option} when interacted with.""",

    """Spot a {size_option} {object_option} {position_option}. It's {color_option}, {shape_option}, constructed of {material_option}, features a {texture_option} {surface_option}, has a weight of {weight_option}, and {deformation_option} upon manipulation.""",

    """A {material_option} {object_option} resides {position_option}. It's {color_option}, {size_option}, {shape_option} in shape, with a {texture_option} {surface_option}. The object weighs {weight_option} and {deformation_option} when touched."""
]

# option_lists = {}
# for key in descriptions.keys():
#     option_lists[f"{key}_option"] = descriptions[key][:]

# prompt = prompt_templates[0].format(**option_lists)

# print("prompt:",prompt)
# description = {
#     "Move":{
#         "rubikscube": "<Description> At the center of the table, there is a colorful cubic object called rubikscube made of plastic with a smooth surface, its size is medium, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of rubikscube.",

#         "tomato": "<Description> At the center of the table, there is a red and white cylindrical object called tomato_soup_can made of metal with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of tomato_soup_can.",

#         "soap": "<Description> At the center of the table, there is a blue rectangular object called soap made of soap with a smooth surface, its size is small, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of soap.", 

#         "plastic_apple": "<Description> At the center of the table, there is a red spherical object called plastic_apple made of plastic with a smooth surface, its size is medium, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of plastic_apple.", 

#         "two_color_hammer": "<Description> At the center of the table, there is a white and black cylindrical object called two_color_hammer made of metal with a textured surface, its size is medium, weight is heavy, and it may have no deformation when interacted with. <Instruction> Move to the place of two_color_hammer.", 

#         "sugar_box": "<Description> At the center of the table, there is a yellow and white rectangular object called sugar_box made of cardboard with a smooth surface, its size is medium, weight is light, and it may have medium deformation when interacted with. <Instruction> Move to the place of sugar_box.", 

#         "shampoo": "<Description> At the center of the table, there is a white and blue cylindrical object called shampoo made of plastic with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of shampoo.", 

#         "magic_clean": "<Description> At the center of the table, there is a green rectangular object called magic_clean made of plastic with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of magic_clean.",
#     },
#     "Grasp":{
#         "rubikscube": "<Description> The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of rubikscube.",

#         "tomato": "<Description> The tomato_soup_can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of tomato_soup_can.",

#         "soap": "<Description> The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of soap.", 

#         "plastic_apple": "<Description> The plastic_apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of plastic_apple.", 

#         "two_color_hammer": "<Description> The two_color_hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation when been grasped. <Instruction> Grasp the object of two_color_hammer.", 

#         "sugar_box": "<Description> The sugar_box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation when been grasped. <Instruction> Grasp the object of sugar_box.", 

#         "shampoo": "<Description> The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of shampoo.", 

#         "magic_clean": "<Description> The magic_clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of magic_clean.",
#     },
#     "Bring":{
#         "rubikscube": "<Description> The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need medium force to grasp when been lifted. <Instruction> Lift the object of rubikscube.",

#         "tomato": "<Description> The tomato_soup_can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of tomato_soup_can.",

#         "soap": "<Description> The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation and need medium force to grasp when been lifted. <Instruction> Lift the object of soap.", 

#         "plastic_apple": "<Description> The plastic_apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need large force to grasp when been lifted. <Instruction> Lift the object of plastic_apple.", 

#         "two_color_hammer": "<Description> The two_color_hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation and need large force to grasp when been lifted. <Instruction> Lift the object of two_color_hammer.", 

#         "sugar_box": "<Description> The sugar_box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation and need light force to grasp when been lifted. <Instruction> Lift the object of sugar_box.", 

#         "shampoo": "<Description> The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of shampoo.", 

#         "magic_clean": "<Description> The magic_clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of magic_clean.",
#     }
# }

description = {
    "Move":{
        "RubiksCube": "<Description> At the center of the table, there is a colorful cubic object called rubikscube made of plastic with a smooth surface, its size is medium, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of rubikscube.",

        "tomato": "<Description> At the center of the table, there is a red and white cylindrical object called tomato_soup_can made of metal with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of tomato_soup_can.",

        "soap": "<Description> At the center of the table, there is a blue rectangular object called soap made of soap with a smooth surface, its size is small, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of soap.", 

        "plastic_apple": "<Description> At the center of the table, there is a red spherical object called plastic_apple made of plastic with a smooth surface, its size is medium, weight is light, and it may have no deformation when interacted with. <Instruction> Move to the place of plastic_apple.", 

        "two_color_hammer": "<Description> At the center of the table, there is a white and black cylindrical object called two_color_hammer made of metal with a textured surface, its size is medium, weight is heavy, and it may have no deformation when interacted with. <Instruction> Move to the place of two_color_hammer.", 

        "sugar_box": "<Description> At the center of the table, there is a yellow and white rectangular object called sugar_box made of cardboard with a smooth surface, its size is medium, weight is light, and it may have medium deformation when interacted with. <Instruction> Move to the place of sugar_box.", 

        "shampoo": "<Description> At the center of the table, there is a white and blue cylindrical object called shampoo made of plastic with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of shampoo.", 

        "magic_clean": "<Description> At the center of the table, there is a green rectangular object called magic_clean made of plastic with a smooth surface, its size is medium, weight is heavy, and it may have slight deformation when interacted with. <Instruction> Move to the place of magic_clean.",
    },
    "Grasp":{
        "RubiksCube": "<Description> The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of rubikscube.",

        "tomato": "<Description> The tomato_soup_can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of tomato_soup_can.",

        "soap": "<Description> The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of soap.", 

        "plastic_apple": "<Description> The plastic_apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. <Instruction> Grasp the object of plastic_apple.", 

        "two_color_hammer": "<Description> The two_color_hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation when been grasped. <Instruction> Grasp the object of two_color_hammer.", 

        "sugar_box": "<Description> The sugar_box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation when been grasped. <Instruction> Grasp the object of sugar_box.", 

        "shampoo": "<Description> The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of shampoo.", 

        "magic_clean": "<Description> The magic_clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. <Instruction> Grasp the object of magic_clean.",
    },
    "Bring":{
        "RubiksCube": "<Description> The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need medium force to grasp when been lifted. <Instruction> Lift the object of rubikscube.",

        "tomato": "<Description> The tomato_soup_can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of tomato_soup_can.",

        "soap": "<Description> The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation and need medium force to grasp when been lifted. <Instruction> Lift the object of soap.", 

        "plastic_apple": "<Description> The plastic_apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need large force to grasp when been lifted. <Instruction> Lift the object of plastic_apple.", 

        "two_color_hammer": "<Description> The two_color_hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation and need large force to grasp when been lifted. <Instruction> Lift the object of two_color_hammer.", 

        "sugar_box": "<Description> The sugar_box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation and need light force to grasp when been lifted. <Instruction> Lift the object of sugar_box.", 

        "shampoo": "<Description> The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of shampoo.", 

        "magic_clean": "<Description> The magic_clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. <Instruction> Lift the object of magic_clean.",
    }
}

description = {
    "Move":{
        "RubiksCube": "The robot arm is not contact the rubikscube.",
        "plastic_apple": "The robot arm is not contact the plastic_apple.",
        "TomatoSoupCan": "The robot arm is not contact the tomato_soup_can.",
        "soap": "The robot arm is not contact the soap.", 
        "two_color_hammer": "The robot arm is not contact the two_color_hammer.", 
        "sugar_box": "The robot arm is not contact the sugar_box.", 
        "shampoo": "The robot arm is not contact the shampoo.", 
        "magic_clean": "The robot arm is not contact the magic_clean.",
    },
    "Grasp":{
        "RubiksCube": "The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. The robot arm is grasping the rubikscube.",
        "plastic_apple": "The plastic_apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation when been grasped. The robot arm is grasping the plastic_apple.",
        "TomatoSoupCan": "The tomato_soup_can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. The robot arm is grasping the tomato_soup_can.",
        "soap": "The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation when been grasped. The robot arm is grasping the soap.", 
        "two_color_hammer": "The two_color_hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation when been grasped. The robot arm is grasping the two_color_hammer.", 
        "sugar_box": "The sugar_box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation when been grasped. The robot arm is grasping the sugar_box.", 
        "shampoo": "The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. The robot arm is grasping the shampoo.", 
        "magic_clean": "The magic_clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation when been grasped. The robot arm is grasping the magic_clean.",
    },
    "Bring":{
        "RubiksCube": "The rubikscube is a colorful cubic object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need medium force to grasp when been lifted. The robot arm is lifting the rubikscube.",
        "plastic_apple": "The plastic_apple is a red spherical object made of plastic with a smooth surface, its size is medium, weight is light, and it has no deformation and need large force to grasp when been lifted. The robot arm is lifting the plastic_apple.", 
        "TomatoSoupCan": "The tomato_soup_can is a red and white cylindrical object made of metal with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. The robot arm is lifting the tomato_soup_can.",
        "soap": "The soap is a blue rectangular object made of soap with a smooth surface, its size is small, weight is light, and it has no deformation and need medium force to grasp when been lifted. The robot arm is lifting the soap.",  
        "two_color_hammer": "The two_color_hammer is a white and black cylindrical object made of metal with a textured surface, its size is medium, weight is heavy, and it has no deformation and need large force to grasp when been lifted. The robot arm is lifting the two_color_hammer.", 
        "sugar_box": "The sugar_box is a yellow and white rectangular object made of cardboard with a smooth surface, its size is medium, weight is light, and it has medium deformation and need light force to grasp when been lifted. The robot arm is lifting the sugar_box.", 
        "shampoo": "The shampoo is a white and blue cylindrical object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. The robot arm is lifting the shampoo.", 
        "magic_clean": "The magic_clean is a green rectangular object made of plastic with a smooth surface, its size is medium, weight is heavy, and it has slight deformation and need large force to grasp when been lifted. The robot arm is lifting the magic_clean.",
    }
}

@DATA_PROCESSER_REGISTRY.register("DescriptionPoselt")
class DescriptionPoselt(BaseDataProcessor):
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

        case_name = json_path.split("/")[-2]
        object_name = case_name.split('_')[0]
        # root_path = get_dirpath_from_key(json_path, "raw_meta")
        # frame_infos = meta_info["frame_info"]
        # window_set = []
        for idx in tqdm(meta_info["frame_info"], colour='green', desc=f'PID[{os.getpid()}]'):
            if self.suffix != None:
                meta_info["frame_info"][idx][f"instruction_{self.suffix}"] = f"Grasp the {object_name}"
            else:
                meta_info["frame_info"][idx]["instruction"] = f"Grasp the {object_name}"
            # meta_info["frame_info"][idx]["label"] = labels["Move"][object_name]
            # meta_info["frame_info"][idx]["action"] = "move"
            # meta_info["frame_info"][idx]["contact_state"] = "no-contact"
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