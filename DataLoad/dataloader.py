import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
from utils.data_func import parse_xml


class PedestrianDataset(Dataset):
    def __init__(self, image_folder, xml_folder, transform=None, target_size=(224, 224)):
        """
        初始化数据集。

        Args:
            image_folder (str): 存储图像的文件夹路径。
            xml_folder (str): 存储 XML 标注文件的文件夹路径。
            transform (callable, optional): 用于图像的转换（如数据增强）。
        """
        self.image_folder = image_folder
        self.xml_folder = xml_folder
        self.transform = transform
        self.target_size = target_size

        # 获取所有图像文件的列表
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.jpeg')]
    
    def __len__(self):
        """返回数据集的大小。"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """根据索引返回单个数据样本。"""
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 将图像调整到目标尺寸
        image = image.resize(self.target_size)
        
        # 加载对应的 XML 文件
        xml_name = image_name.replace('.jpg', '.xml').replace('.jpeg', '.xml')
        xml_path = os.path.join(self.xml_folder, xml_name)
        
        boxes, labels = parse_xml(xml_file=xml_path)

        # 转换为 PyTorch 张量
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # 可选的图像变换
        if self.transform:
            image = self.transform(image)

        return image, boxes, labels


# 对每个batch的批处理，只考虑一个batch中的数据
def collate_fn(batch):
    images, boxes, labels = zip(*batch)

    # 确定最大目标框数量
    max_num_boxes = max([len(b) for b in boxes])

    # 填充目标框
    padded_boxes = []
    padded_labels = []
    for i, b in enumerate(boxes):
        # 填充目标框
        padded_boxes.append(np.pad(b, ((0, max_num_boxes - len(b)), (0, 0)), mode='constant', constant_values=0))
        # 填充标签
        label_len = len(b)
        padded_label = np.pad(labels[i].numpy(), (0, max_num_boxes - label_len), mode='constant', constant_values=0)
        padded_labels.append(padded_label)

    # 转换为张量
    padded_boxes = torch.tensor(padded_boxes, dtype=torch.float32)
    padded_labels = torch.tensor(padded_labels, dtype=torch.long)

    # 堆叠图像
    images = torch.stack(images, 0)

    return images, padded_boxes, padded_labels


if __name__ == "__main__":
    from utils.read_cfg import load_yaml_as_namespace
    from utils.func import get_absolute_path

    file_path = "utils/config.yaml"                             # 替换成你的YAML文件路径
    args = load_yaml_as_namespace(get_absolute_path(file_path))

    image_folder = args.image_folder
    xml_folder = args.xml_folder

    # 创建数据集
    dataset = PedestrianDataset(image_folder, xml_folder, transform=ToTensor())

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    # 使用 DataLoader 迭代数据
    for images, boxes, labels in dataloader:
        print(f"Images batch shape: {images.shape}")
        print(f"Boxes: {boxes}")
        print(f"Labels: {labels}")
