import os
import torch
from PIL import Image

import sys
sys.path.append("F:\Lwm_Code\Visual")

from DataLoad.image_augment import basic_augment
from utils.data_func import parse_xml
from utils.read_cfg import load_yaml_as_namespace
from utils.func import get_absolute_path, draw_boxes_pil


if __name__ == "__main__":
    file_path = "utils/config.yaml"                             # 替换成你的YAML文件路径
    args = load_yaml_as_namespace(get_absolute_path(file_path))

    image_folder = args.image_folder
    xml_folder = args.xml_folder
    target_size = args.target_size
    image_save_path = args.image_save_path
    

    for i, image_name in enumerate(os.listdir(image_folder)[:10]):
        image_path = os.path.join(image_folder, image_name)

        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 加载对应的 XML 文件
        xml_name = image_name.replace('.jpg', '.xml').replace('.jpeg', '.xml')
        xml_path = os.path.join(xml_folder, xml_name)
        
        boxes, labels = parse_xml(xml_file=xml_path)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        image, boxes = basic_augment(args, image, boxes, target_size)

        image = draw_boxes_pil(image, boxes)

        image.save(os.path.join(image_save_path, f"image_{i}.jpg"))