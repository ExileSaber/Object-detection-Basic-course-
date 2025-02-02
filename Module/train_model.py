import torch
import torch.nn as nn
import torch.nn.functional as F
from Module.simple_model import SimpleDetector
from DataLoad.dataloader import get_dataloader
from DataLoad.image_augment import basic_augment


def start_train(args):
    model = SimpleDetector(num_classes=1)

    if args.augment == "basic":
        augment = basic_augment
    else:
        augment = None

    train_loader, val_loader, test_loader = get_dataloader(args, augment)

    # 损失函数
    classification_loss_fn = nn.CrossEntropyLoss()
    regression_loss_fn = nn.SmoothL1Loss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for images, boxes, labels, masks in train_loader:
            # 前向传播
            class_logits, bbox_preds = model(images)

            class_logits = class_logits.squeeze(-1)
            
            # 计算损失
            # print(class_logits.shape)  # 应该是 [batch_size, num_boxes, num_classes]
            # print(labels.shape)        # 应该是 [batch_size, num_boxes]
            cls_loss = classification_loss_fn(class_logits[masks], labels[masks])
            reg_loss = regression_loss_fn(bbox_preds[masks], boxes[masks])
            loss = cls_loss + reg_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
