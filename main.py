from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from DataLoad.dataloader import PedestrianDataset, collate_fn
from utils.read_cfg import load_yaml_as_namespace
from utils.func import get_absolute_path


if __name__ == "__main__":
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
