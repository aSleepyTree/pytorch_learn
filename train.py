import torch.optim
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from module import *
from torch.utils.data import DataLoader
# 从module中导入网络模型

# 准备数据集
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 获取数据及长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print('训练数据集长度为:{}'.format(train_data_size))
print('测试数据集长度为:{}'.format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# 创建网络模型
work = Work()

# 选择损失函数为交叉熵
loss_fn = nn.CrossEntropyLoss()

# 选择优化器为随机梯度下降
leaning_rate = 1e-2
optimizer = torch.optim.SGD(work.parameters(), lr=leaning_rate)

# 设置网络参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 10


# 添加tensorboard
writer = SummaryWriter('./logs_maxpool')
for i in range(epoch):
    print("----------------第{}轮训练开始------------".format(1 + i))
    for data in train_dataloader:
        imgs, targets = data
        outputs = work(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if not total_train_step % 100:
            print('训练次数：{}，Loss：{}'.format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = work(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
        print("整体测试集上的Loss：{}".format(total_test_loss))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        total_test_step += 1

    writer.close()