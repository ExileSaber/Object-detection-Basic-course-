import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import ToTensor
from utils.data_func import parse_xml
from DataLoad.image_augment import basic_augment


MAX_NUM_BOXES = 49


class PedestrianDataset(Dataset):
    def __init__(self, args, image_folder, xml_folder, transform=None, target_size=(224, 224), augmentations=None):
        self.args = args
        self.image_folder = image_folder
        self.xml_folder = xml_folder
        self.transform = transform
        self.target_size = target_size
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.jpeg')]
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)

        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 加载对应的 XML 文件
        xml_name = image_name.replace('.jpg', '.xml').replace('.jpeg', '.xml')
        xml_path = os.path.join(self.xml_folder, xml_name)
        
        boxes, labels = parse_xml(xml_file=xml_path)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.augmentations:
            image, boxes, labels = self.augmentations(self.args, image, boxes, labels, self.target_size)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels


# 自定义collate_fn来处理不同数量的目标框
def collate_fn(batch):
    images, boxes, labels = zip(*batch)

    padded_boxes = []
    padded_labels = []
    masks = []  # 用于标记有效框的 mask

    for i, b in enumerate(boxes):
        num_boxes = min(len(b), MAX_NUM_BOXES)  # 限制最大框数

        # 1. 计算填充长度
        pad_len = max(0, MAX_NUM_BOXES - num_boxes)

        # 2. 填充边界框，使用 [-1, -1, -1, -1] 作为无效框
        padded_box = np.pad(b, ((0, pad_len), (0, 0)), mode='constant', constant_values=-1)
        padded_boxes.append(padded_box)

        # 3. 填充类别标签，使用 -1 作为无效类别
        padded_label = np.pad(labels[i].numpy(), (0, pad_len), mode='constant', constant_values=-1)
        padded_labels.append(padded_label)

        # 4. 生成有效 mask（真实框为 1，填充框为 0）
        mask = np.pad(np.ones(num_boxes), (0, pad_len), mode='constant', constant_values=0)
        masks.append(mask)

    # 转换为 PyTorch 张量
    padded_boxes = torch.tensor(np.array(padded_boxes), dtype=torch.float32)  # 形状: (batch_size, max_num_boxes, 4)
    padded_labels = torch.tensor(np.array(padded_labels), dtype=torch.long)   # 形状: (batch_size, max_num_boxes)
    masks = torch.tensor(np.array(masks), dtype=torch.bool)                   # 形状: (batch_size, max_num_boxes)

    # 处理图像
    images = torch.stack(images, 0)  # 形状: (batch_size, C, H, W)

    return images, padded_boxes, padded_labels, masks


def get_dataloader(args, augmentations=None):
    batch_size = args.batch_size
    # 创建数据集
    dataset = PedestrianDataset(args, args.image_folder, args.xml_folder, transform=ToTensor(), augmentations=augmentations)

    # 计算划分的大小
    total_size = len(dataset)
    train_size = int(args.train_rate * total_size)
    val_size = int(args.val_rate * total_size)
    test_size = total_size - train_size - val_size

    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return train_loader, val_loader, test_loader