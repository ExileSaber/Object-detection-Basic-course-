import torch
import random
from torchvision.transforms import functional as F
from torchvision.transforms import RandomCrop

# 增强方法：常见的增强方法
def basic_augment(args, image, boxes, labels, target_size):
    width, height = image.size  # (width, height)

    # 图像缩放到目标尺寸
    image = image.resize(target_size)

    # 同步缩放框
    scale_x = target_size[0] / width
    scale_y = target_size[1] / height
    boxes *= torch.tensor([scale_x, scale_y, scale_x, scale_y])

    width, height = target_size

    # 随机水平翻转
    if random.random() < args.hflip:
        image = F.hflip(image)
        boxes = torch.tensor([[width - b[2], b[1], width - b[0], b[3]] for b in boxes])  # 更新框坐标

    # 随机旋转
    if random.random() < args.rotate:
        angle = random.choice([90, 180, 270])  # 随机选择旋转角度
        image = F.rotate(image, angle)

        if angle == 90:
            boxes = torch.tensor([[b[1], width - b[2], b[3], width - b[0]] for b in boxes])
        elif angle == 180:
            boxes = torch.tensor([[width - b[2], height - b[3], width - b[0], height - b[1]] for b in boxes])
        elif angle == 270:
            boxes = torch.tensor([[height - b[3], b[0], height - b[1], b[2]] for b in boxes])

        # 交换宽高
        width, height = height, width

    # 随机色彩抖动
    if random.random() < args.color:
        image = F.adjust_brightness(image, random.uniform(0.7, 1.3))
        image = F.adjust_contrast(image, random.uniform(0.7, 1.3))
        image = F.adjust_saturation(image, random.uniform(0.7, 1.3))

    # 随机裁剪
    if random.random() < args.crop:
        crop_size = (random.randint(150, height), random.randint(150, width))
        crop = RandomCrop(crop_size)
        i, j, h, w = crop.get_params(image, crop_size)
        image = F.crop(image, i, j, h, w)

        # 更新目标框坐标，裁剪后调整框的位置
        boxes[:, [0, 2]] -= j  # x_min, x_max
        boxes[:, [1, 3]] -= i  # y_min, y_max

        # 处理越界情况，确保框仍然在裁剪后的图像内
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)  # 限制 x 坐标
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)  # 限制 y 坐标

        # 只保留有效的框（w > 0 and h > 0）
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]
        labels = labels[valid_boxes]

        # 填充图像到目标大小
        padding = (0, 0, target_size[0] - w, target_size[1] - h)
        image = F.pad(image, padding, fill=0, padding_mode='constant')

    

    return image, boxes, labels
