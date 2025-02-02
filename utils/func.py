import os
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
from utils.data_func import parse_xml

def get_absolute_path(path):
    current_directory = os.getcwd()
    # print("当前工作目录的绝对路径:", current_directory)
    path = os.path.join(current_directory, path)
    return path


def draw_boxes_pil(image, boxes, save_path=None):
    """
    使用 PIL 在图像上绘制目标框，并保存到本地。

    Args:
        image (Tensor or PIL.Image): 输入的图像（Tensor 格式或 PIL 格式）。
        boxes (Tensor): 目标框，形状为 (N, 4)，格式为 [x_min, y_min, x_max, y_max]。
        save_path (str): 存储路径。
    """
    # 如果 image 是 Tensor，需要转换为 PIL Image
    if isinstance(image, torch.Tensor):
        image = ToPILImage()(image)

    # 创建绘图对象
    draw = ImageDraw.Draw(image)

    # 绘制目标框
    for box in boxes:
        x_min, y_min, x_max, y_max = box.tolist()
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    # 保存图像
    if save_path:
        image.save(save_path)
        print(f"保存图像: {save_path}")
    
    return image