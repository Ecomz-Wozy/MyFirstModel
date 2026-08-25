import torch
import torchvision
import sys
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 下载数据并计算其长度
train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
len_train_data = len(train_data)
len_test_data = len(test_data)

# 准备数据载体
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)

# 定义模型
class myFirstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

mfm = myFirstModel()
# 利用GPU加速训练模型
if torch.cuda.is_available():
    mfm = mfm.cuda()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 定义优化器和学习率
learning_rate = 0.01
optimizer = torch.optim.SGD(mfm.parameters(), lr=learning_rate)

writer = SummaryWriter("logs")
epoch = 100
label_max_accurate = 1
max_accuracy = 0
label_min_loss = 1
min_loss = sys.float_info.max

# 训练并测试模型 epoch 轮
for i in range(epoch):
    # 开始训练
    print("第{}轮训练开始".format(i+1))
    mfm.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = mfm(imgs) # 模型预测
        loss = loss_fn(outputs, targets) # 计算错误程度 Loss

        # 清空旧梯度
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 根据梯度修改参数
        optimizer.step()

    # 测试开始
    print("第{}轮测试开始".format(i+1))
    mfm.eval()
    accurate_case = 0
    total_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = mfm(imgs)
            total_loss += loss_fn(outputs, targets)
            accurate_case += (outputs.argmax(1) == targets).sum()

    print("测试集上整体损失为：{}".format(total_loss))
    total_accuracy = accurate_case/len_test_data
    print("正确率为：{}".format(total_accuracy))
    writer.add_scalar("test_loss", total_loss, i)
    writer.add_scalar("test_accuracy", accurate_case / len_test_data, i)
    if total_loss < min_loss:
        min_loss = total_loss
        label_min_loss = i + 1
    if max_accuracy < total_accuracy:
        max_accuracy = total_accuracy
        label_max_accurate = i + 1

    # 保存模型
    torch.save(mfm, "./model_version/mfm_v{}.pth".format(i+1))
    print("模型已保存")

writer.close()
print("正确率最高的模型编号是：{}，正确率为；{}".format(label_max_accurate, max_accuracy))
print("损失函数值最小的模型编号是：{}，损失函数值为：{}".format(label_min_loss, min_loss))