# myFirstModel — CIFAR10 卷积神经网络训练

我的第一个 PyTorch 深度学习项目：用 `torch.nn.Sequential` 搭建一个简单 CNN，在 CIFAR10 数据集上训练、测试，并支持 GPU 加速。

## 功能

- 用 `nn.Sequential` 搭建 3 层卷积网络（3×Conv2d + 3×MaxPool2d + Flatten + 2×Linear）
- 自动检测 CUDA，模型和数据搬到 GPU 加速
- 用 TensorBoard 记录每轮训练/测试的损失和准确率
- 每轮训练结束自动保存一个模型版本到 `model_version/`

## 环境要求

- Python 3.x
- PyTorch（GPU 版推荐，如 `torch 2.11.0+cu128`）
- torchvision
- tensorboard

本项目在 conda 的 `gpu` 环境（Python 3.10 / torch 2.11 + CUDA / RTX 5060 Laptop）下开发调试。

## 数据集准备（重要）

脚本里设置的是 `download=False`，**不会自动下载数据**。运行前需要手动准备好 CIFAR10 数据集，目录结构必须是：

```
<运行目录>/
  data/
    cifar-10-batches-py/
      data_batch_1 ~ data_batch_5
      test_batch
```

两种准备方式：

1. **让 torchvision 下载并自动解压**（需要联网，跑一次即可）：
   ```python
   import torchvision
   torchvision.datasets.CIFAR10("./data", train=True, download=True)
   torchvision.datasets.CIFAR10("./data", train=False, download=True)
   ```
   下载后就能用 `download=False` 直接加载。

2. **手动下载解压**：从官网下载 [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)，解压后把 `cifar-10-batches-py` 整个文件夹放到 `data/` 下。

> ⚠️ 注意：`./data` 是相对路径，**从哪个目录运行命令，就在那个目录下找 `data/`**。

## 运行

在包含 `data/` 的目录下执行：

```bash
python myFirstModel.py
```

默认训练 10 个 epoch，每轮结束在测试集上评估损失与准确率，曲线写入 `logs/`，模型保存到 `model_version/`，最后打印最优轮次信息。

## 查看训练曲线

```bash
tensorboard --logdir logs
```

浏览器打开 http://localhost:6006

## 代码结构

| 部分 | 说明 |
|---|---|
| 模型 `myFirstModel` | `nn.Sequential`：Conv2d(3,32)→MaxPool→Conv2d(32,32)→MaxPool→Conv2d(32,64)→MaxPool→Flatten→Linear(1024,64)→Linear(64,10) |
| 损失函数 | `nn.CrossEntropyLoss()` |
| 优化器 | SGD，学习率 0.01 |
| 训练配置 | batch_size=64，10 个 epoch |

