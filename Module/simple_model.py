import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDetector(nn.Module):
    def __init__(self, num_classes=2):  # 假设2类，1类为目标，1类为背景
        super(SimpleDetector, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 输入通道3，输出通道16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 下采样

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 分类头 (用于检测是否存在目标)
        self.classification_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # 回归头 (用于边界框预测)
        self.regression_head = nn.Conv2d(64, 4, kernel_size=1)  # 4维坐标 (x_min, y_min, x_max, y_max)
    
    def forward(self, x):
        # 提取特征
        x = self.features(x)
        # 分类
        class_logits = self.classification_head(x)
        # 边界框回归
        bbox_preds = self.regression_head(x)

        # 展平维度
        class_logits = class_logits.permute(0, 2, 3, 1).contiguous()  # [N, H, W, num_classes]
        bbox_preds = bbox_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, 4]

        # 展开为 [N, -1, num_classes] 和 [N, -1, 4]
        class_logits = class_logits.view(class_logits.size(0), -1, class_logits.size(-1))
        bbox_preds = bbox_preds.view(bbox_preds.size(0), -1, 4)

        return class_logits, bbox_preds


# 测试模型
if __name__ == "__main__":
    # 创建模型
    model = SimpleDetector(num_classes=2)

    # 创建一个随机输入 (Batch Size, Channels, Height, Width)
    input_tensor = torch.randn(1, 3, 224, 224)

    # 前向传播
    class_logits, bbox_preds = model(input_tensor)

    print("分类输出形状:", class_logits.shape)  # [N, H*W, num_classes], [1, 784, 2]
    print("回归输出形状:", bbox_preds.shape)    # [N, H*W, 4], [1, 784, 4]
