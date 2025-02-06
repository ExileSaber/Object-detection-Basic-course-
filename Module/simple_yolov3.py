import torch
import torch.nn as nn
import torch.nn.functional as F

# 简化版 YOLOv3 网络结构
class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        
        # 1. Backbone: Darknet53 简化版
        self.conv1 = self._conv_block(3, 32)
        self.conv2 = self._conv_block(32, 64)
        self.conv3 = self._conv_block(64, 128)
        self.conv4 = self._conv_block(128, 256)
        self.conv5 = self._conv_block(256, 512)
        self.conv6 = self._conv_block(512, 1024)

        # 2. Detection head (3个尺度上的预测)
        self.detect1 = self._detect_block(1024, num_classes)
        self.detect2 = self._detect_block(512, num_classes)
        self.detect3 = self._detect_block(256, num_classes)

    def _conv_block(self, in_channels, out_channels, stride=1, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def _detect_block(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_channels, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # Backbone 前向传播
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        # Detection Heads
        out1 = self.detect1(x6)  # 最细的尺度
        out2 = self.detect2(x5)  # 中等尺度
        out3 = self.detect3(x4)  # 最粗的尺度

        return out1, out2, out3


def yolo_loss(preds, targets, num_classes, anchors, device):
    """
    preds: 模型的输出（包括类别和边界框的预测），形状为 (batch_size, grid_size, grid_size, anchors * (5 + num_classes))
    targets: 真实标签
    num_classes: 类别数量
    anchors: 锚框大小
    """
    loss_conf = 0
    loss_cls = 0
    loss_loc = 0
    num_anchors = len(anchors)

    # 处理每个尺度的输出
    for pred, target in zip(preds, targets):
        # 解码预测结果
        pred = pred.view(-1, num_anchors, 5 + num_classes)
        
        # 计算分类损失
        cls_loss = F.cross_entropy(pred[..., 5:], target[..., 5:])
        
        # 计算边界框损失
        loc_loss = F.mse_loss(pred[..., :4], target[..., :4])

        # 计算置信度损失
        conf_loss = F.binary_cross_entropy_with_logits(pred[..., 4], target[..., 4])

        loss_cls += cls_loss
        loss_loc += loc_loss
        loss_conf += conf_loss

    total_loss = loss_cls + loss_loc + loss_conf
    return total_loss
