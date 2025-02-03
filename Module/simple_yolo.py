import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=20, grid_size=7, bbox_attrs=5):
        """
        初始化一个简单的YOLO风格目标检测网络。

        Args:
            num_classes (int): 类别数量。
            grid_size (int): 每个边的网格数（通常是7x7）。
            bbox_attrs (int): 每个边界框预测的属性数量（x, y, w, h, confidence）。
        """
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.bbox_attrs = bbox_attrs

        # Backbone：简单的卷积层，提取特征
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

        # 检测头部：输出每个网格的预测
        # 输出维度为 (B, S, S, B * (5 + C)) -> (B, 7, 7, 2 * (5 + num_classes))
        self.detector = nn.Conv2d(512, self.grid_size * self.grid_size * self.bbox_attrs * 2, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = self.detector(x)
        x = x.view(batch_size, self.grid_size, self.grid_size, self.bbox_attrs * 2)
        return x

# 测试网络结构
if __name__ == "__main__":
    model = SimpleYOLO(num_classes=3)  # 假设只有3类
    test_input = torch.randn(2, 3, 224, 224)  # Batch size 2, 3通道 (RGB), 224x224 图像
    test_output = model(test_input)
    print("输出维度:", test_output.shape)  # 期望输出 (2, 7, 7, 10) -> 每个网格预测2个框，每个框有5个参数 + 3个类别
