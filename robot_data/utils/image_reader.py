import os
import cv2
from PIL import Image
import numpy as np

def get_cur_image_dir(image_dir, idx):
    if isinstance(image_dir, list) or isinstance(image_dir, tuple):
        assert idx < len(image_dir)
        return image_dir[idx]
    return image_dir

class ImageReader(object):
    def __init__(self, image_dir, color_mode, memcached=None):
        super(ImageReader, self).__init__()
        if image_dir == "/" or image_dir == "//":
            image_dir = ""
        self.image_dir = image_dir
        self.color_mode = color_mode

    def image_directory(self):
        return self.image_dir

    def image_color(self):
        return self.color_mode

    def hash_filename(self, filename):
        import hashlib

        md5 = hashlib.md5()
        md5.update(filename.encode("utf-8"))
        hash_filename = md5.hexdigest()
        return hash_filename

    def read(self, filename):
        raise NotImplementedError

    def __call__(self, filename, image_dir_idx=0):
        if filename.startswith("//"):
            filename = filename[1:]
        image_dir = get_cur_image_dir(self.image_dir, image_dir_idx)
        filename = os.path.join(image_dir, filename)
        img = self.read(filename)
        return img


class FileSystemPILReader(ImageReader):
    def __init__(self, image_dir, color_mode, memcached=None):
        super(FileSystemPILReader, self).__init__(image_dir, color_mode, memcached)
        assert color_mode == "RGB", "only RGB mode supported for pillow for now"

    def fake_image(self, *size):
        if len(size) == 0:
            size = (512, 512, 3)
        return Image.new(self.color_mode, size)

    def fs_read(self, filename):
        assert os.path.exists(filename), filename
        img = Image.open(filename).convert(self.color_mode)
        return img

