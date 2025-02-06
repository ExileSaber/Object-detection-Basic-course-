import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from Module.simple_model import SimpleDetector
from Module.simple_yolov3 import YOLOv3, yolo_loss
from DataLoad.dataloader import get_dataloader
from DataLoad.image_augment import basic_augment


# 计算IoU损失
def iou_loss(pred_boxes, true_boxes):
    # 预测框的坐标
    x1_pred, y1_pred, x2_pred, y2_pred = pred_boxes.unbind(1)
    # 真实框的坐标
    x1_true, y1_true, x2_true, y2_true = true_boxes.unbind(1)

    # 计算交集区域
    inter_x1 = torch.max(x1_pred, x1_true)
    inter_y1 = torch.max(y1_pred, y1_true)
    inter_x2 = torch.min(x2_pred, x2_true)
    inter_y2 = torch.min(y2_pred, y2_true)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算预测框和真实框的面积
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    true_area = (x2_true - x1_true) * (y2_true - y1_true)

    # 计算IoU
    union_area = pred_area + true_area - inter_area
    iou = inter_area / union_area

    # IoU损失，1 - IoU
    return 1 - iou

def simple_model_train_iouLoss(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 修正 num_classes=2
    model = SimpleDetector(num_classes=args.num_classes).to(device)

    if args.augment == "basic":
        augment = basic_augment
    else:
        augment = None

    train_loader, val_loader, test_loader = get_dataloader(args, augment)

    classification_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 定义线性学习率调度器
    def lr_lambda(epoch):
        # 线性减少学习率，从初始值减少到0
        lr = args.lr * (0.1 ** (epoch // args.change_lr_step))
        return lr

    scheduler = LambdaLR(optimizer, lr_lambda)

    for epoch in range(args.epochs):
        model.train()
        for images, boxes, labels, masks in train_loader:
            images, boxes, labels, masks = images.to(device), boxes.to(device), labels.to(device), masks.to(device)

            # 前向传播
            class_logits, bbox_preds = model(images)

            # print(class_logits.shape)
            class_logits = class_logits.view(class_logits.size(0), -1, class_logits.size(-1))
            bbox_preds = bbox_preds.view(bbox_preds.size(0), -1, 4)

            # 归一化 bbox
            # bbox_preds = torch.sigmoid(bbox_preds)  # 确保输出 bbox 在 [0,1] 范围

            # 计算损失
            if masks.sum() == 0:
                continue  # 避免 NaN

            cls_loss = classification_loss_fn(class_logits[masks], labels[masks])
            iou_loss_value = iou_loss(bbox_preds[masks], boxes[masks]).mean()

            loss = cls_loss + iou_loss_value

            if args.show_detail_loss and epoch % args.show_detail_epoch == 0:
                print("------------- 详细内容 -------------")
                print(f"masks: {masks}")
                print(f"真实目标标签: {labels[masks]}")
                print(f"预测目标标签: {class_logits[masks]}")
                print(f"真实包围框: {boxes[masks]}")
                print(f"预测包围框: {bbox_preds[masks]}")
                print(f"cls_loss: {cls_loss}")
                print(f"iou_loss: {iou_loss_value}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

        print(f"Epoch {epoch+1}, Loss: /{loss.item()}")


def simple_model_train_F1Loss(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 修正 num_classes=2
    model = SimpleDetector(num_classes=args.num_classes).to(device)

    if args.augment == "basic":
        augment = basic_augment
    else:
        augment = None

    train_loader, val_loader, test_loader = get_dataloader(args, augment)

    classification_loss_fn = nn.CrossEntropyLoss()
    regression_loss_fn = nn.SmoothL1Loss(reduction='sum')  # 使用 sum 避免 0 损失

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 定义线性学习率调度器
    # 定义线性学习率调度器
    def lr_lambda(epoch):
        # 线性减少学习率，从初始值减少到0
        lr = args.lr * (0.1 ** (epoch // args.change_lr_step))
        return lr

    scheduler = LambdaLR(optimizer, lr_lambda)

    for epoch in range(args.epochs):
        model.train()
        for images, boxes, labels, masks in train_loader:
            images, boxes, labels, masks = images.to(device), boxes.to(device), labels.to(device), masks.to(device)

            # 前向传播
            class_logits, bbox_preds = model(images)

            # print(class_logits.shape)
            class_logits = class_logits.view(class_logits.size(0), -1, class_logits.size(-1))
            bbox_preds = bbox_preds.view(bbox_preds.size(0), -1, 4)

            # 归一化 bbox
            # bbox_preds = torch.sigmoid(bbox_preds)  # 确保输出 bbox 在 [0,1] 范围

            # 计算损失
            if masks.sum() == 0:
                continue  # 避免 NaN

            cls_loss = classification_loss_fn(class_logits[masks], labels[masks])
            reg_loss = regression_loss_fn(bbox_preds[masks], boxes[masks]) / max(1, masks.sum())

            loss = cls_loss + reg_loss

            if args.show_detail_loss and epoch % args.show_detail_epoch == 0:
                print("------------- 详细内容 -------------")
                print(f"masks: {masks}")
                print(f"真实目标标签: {labels[masks]}")
                print(f"预测目标标签: {class_logits[masks]}")
                print(f"真实包围框: {boxes[masks]}")
                print(f"预测包围框: {bbox_preds[masks]}")
                print(f"cls_loss: {cls_loss}")
                print(f"reg_loss: {reg_loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")



def simple_yolov3_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 修正 num_classes=2
    model = SimpleDetector(num_classes=args.num_classes).to(device)

    if args.augment == "basic":
        augment = basic_augment
    else:
        augment = None

    train_loader, val_loader, test_loader = get_dataloader(args, augment)

    # 初始化模型和优化器
    model = YOLOv3(args.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练过程
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, boxes, labels, masks in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = yolo_loss(outputs, targets, args.num_classes, anchors=None, device='cuda')
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader)}")
