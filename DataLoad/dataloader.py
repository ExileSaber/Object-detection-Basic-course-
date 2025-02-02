import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import ToTensor
from utils.data_func import parse_xml
from DataLoad.image_augment import basic_augment


class PedestrianDataset(Dataset):
    def __init__(self, image_folder, xml_folder, transform=None, target_size=(224, 224), augmentations=None):
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
        labels = torch.tensor(labels, dtype=torch.long)

        if self.augmentations:
            image, boxes = self.augmentations(args, image, boxes, self.target_size)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels


# 自定义collate_fn来处理不同数量的目标框
def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    max_num_boxes = max([len(b) for b in boxes])

    padded_boxes = []
    padded_labels = []
    for i, b in enumerate(boxes):
        padded_boxes.append(np.pad(b, ((0, max_num_boxes - len(b)), (0, 0)), mode='constant', constant_values=0))
        label_len = len(b)
        padded_label = np.pad(labels[i].numpy(), (0, max_num_boxes - label_len), mode='constant', constant_values=0)
        padded_labels.append(padded_label)

    padded_boxes = torch.tensor(padded_boxes, dtype=torch.float32)
    padded_labels = torch.tensor(padded_labels, dtype=torch.long)

    images = torch.stack(images, 0)

    return images, padded_boxes, padded_labels


def get_dataloader(args, augmentations=None):
    # 创建数据集
    dataset = PedestrianDataset(args.image_folder, args.xml_folder, transform=ToTensor(), augmentations=augmentations)

    # 计算划分的大小
    total_size = len(dataset)
    train_size = int(args.train_rate * total_size)
    val_size = int(args.val_rate * total_size)
    test_size = total_size - train_size - val_size

    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from utils.read_cfg import load_yaml_as_namespace
    from utils.func import get_absolute_path

    file_path = "utils/config.yaml"
    args = load_yaml_as_namespace(get_absolute_path(file_path))

    image_folder = args.image_folder
    xml_folder = args.xml_folder
    train_rate = args.train_rate
    val_rate = args.val_rate

    train_loader, val_loader, test_loader = get_dataloader(image_folder, xml_folder, train_rate, val_rate, basic_augment)

    # 使用 DataLoader 迭代数据
    for images, boxes, labels in train_loader:
        print(f"Train batch shape: {images.shape}")
        print(f"Boxes: {boxes}")
        print(f"Labels: {labels}")

    for images, boxes, labels in val_loader:
        print(f"Validation batch shape: {images.shape}")
        print(f"Boxes: {boxes}")
        print(f"Labels: {labels}")

    for images, boxes, labels in test_loader:
        print(f"Test batch shape: {images.shape}")
        print(f"Boxes: {boxes}")
        print(f"Labels: {labels}")
