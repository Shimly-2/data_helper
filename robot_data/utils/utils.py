import os
import os.path as osp
import json


def write_txt(files, path, file_name=None, mod="w+"):
    if not path.endswith(".txt"):
        if file_name is not None and file_name.endswith(".txt"):
            path = osp.join(path, file_name)
    if not osp.exists(path) and mod.startswith("a"):
        mod = "w+"
    if not osp.exists(osp.split(path)[0]):
        os.makedirs(osp.split(path)[0])
    with open(path, mode=mod) as f:
        for line in files:
            f.write(line)
            f.write("\n")


def read_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
    results = [line.strip() for line in lines]
    return results


def write_json(files, path, file_name=None, mod="w+"):
    if not path.endswith(".json") or path.endswith(".txt"):
        if file_name is not None and (
            file_name.endswith(".json") or file_name.endswith(".txt")
        ):
            path = osp.join(path, file_name)
    if not osp.exists(path) and mod.startswith("a"):
        mod = "w+"
    if not osp.exists(osp.split(path)[0]):
        os.makedirs(osp.split(path)[0])
    with open(path, mod) as f:
        if isinstance(files, dict):
            json.dump(files, f)
        elif isinstance(files, list):
            for line in files:
                json.dump(line, f)
                f.write("\n")
        else:
            raise NotImplementedError


def read_json(path):
    with open(path, "r") as f:
        lines = f.readlines()
    meta_json = [json.loads(line) for line in lines]
    return meta_json
