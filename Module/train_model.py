import torch
import torch.nn as nn
import torch.nn.functional as F
from Module.simple_model import SimpleDetector
from DataLoad.dataloader import get_dataloader
from DataLoad.image_augment import basic_augment


def start_train(args):
    # 设置设备，检查是否有可用的GPU，如果没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型并将模型迁移到指定的设备
    model = SimpleDetector(num_classes=1).to(device)

    if args.augment == "basic":
        augment = basic_augment
    else:
        augment = None

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloader(args, augment)

    # 损失函数
    classification_loss_fn = nn.CrossEntropyLoss()
    regression_loss_fn = nn.SmoothL1Loss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()  # 训练模式
        for images, boxes, labels, masks in train_loader:
            # 将数据移动到GPU
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # 前向传播
            class_logits, bbox_preds = model(images)

            # 计算损失
            class_logits = class_logits.squeeze(-1)
            cls_loss = classification_loss_fn(class_logits[masks], labels[masks])
            reg_loss = regression_loss_fn(bbox_preds[masks], boxes[masks])
            loss = cls_loss + reg_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印每个epoch的损失
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

