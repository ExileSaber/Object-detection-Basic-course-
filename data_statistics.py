"""
数据统计，主要需要统计每个图像的尺寸、图像后缀、每张图像标签的多少
"""

import os
import numpy as np
from utils.data_func import parse_xml
from PIL import Image
from collections import Counter


def image_analyze(path):
    end_list = []
    size_list = []

    for image_name in os.listdir(path):
        end_list.append(image_name.split(".")[-1])

        image = Image.open(os.path.join(path, image_name)).convert("RGB")
        size_list.append(image.size)

    print("-------------- 后缀统计 --------------\n", Counter(end_list))
    print("-------------- 尺寸统计 --------------\n", Counter(size_list))
    

def label_analyze(path):
    labellen_list = []

    for xml_name in os.listdir(path):
        boxes, labels = parse_xml(os.path.join(path, xml_name))
        labellen_list.append(len(boxes))

    print("-------------- 标签统计 --------------\n", Counter(labellen_list))
    print("-------------- 最大长度 --------------\n", max(labellen_list))


if __name__ == "__main__":
    from utils.read_cfg import load_yaml_as_namespace
    from utils.func import get_absolute_path

    config_path = "utils/config.yaml"
    args = load_yaml_as_namespace(get_absolute_path(config_path))

    # image_path = args.image_folder
    # image_analyze(image_path)

    xml_path = args.xml_folder
    label_analyze(get_absolute_path(xml_path))
