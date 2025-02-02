from DataLoad.dataloader import get_dataloader
from utils.read_cfg import load_yaml_as_namespace
from utils.func import get_absolute_path
from DataLoad.image_augment import basic_augment

if __name__ == "__main__":
    file_path = "utils/config.yaml"                             # YAML文件路径
    args = load_yaml_as_namespace(get_absolute_path(file_path))

    train_loader, val_loader, test_loader = get_dataloader(args, basic_augment)

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
