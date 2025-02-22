# 从零开始的项目流程

## 1. 统计数据集的情况

1. 图像的后缀

2. 图像标签的类型

运行代码 `data_statistics.py`


## 2. 数据预处理

1. 剔除显然的异常数据

2. 各种数据增强

3. 根据任务需求调整数据格式

对应代码 `DataLoad/image_augment.py`，检查效果代码 `Test/test_image_augment.py`

## 3. 数据迭代器

1. 构造数据集，整合数据增强和后续处理

2. 训练验证测试集的划分

对应代码 `DataLoad/dataloader.py`，检查效果代码 `Test/test_dataloader.py`

## 4. 模型搭建

1. 搭建卷积、全连接、池化、正则化等基础网络

2. 学习有效的复杂网络结构并搭建


## 5. 损失函数及优化器

1. 定义优化器和损失函数

2. 动态调整学习率

3. 梯度回传等


## 6. 训练

1. 知道多种训练策略

2. 能够实时观察训练曲线


## 7. 部署

1. 掌握部署到开发板上的能力 