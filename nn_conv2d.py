import torch
from torch import nn
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("/data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
downloader = DataLoader(dataset, batch_size=64)


class Work(nn.Module):
    def __init__(self):
        super(Work, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=
        6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


work = Work()

writer = SummaryWriter("logs")
step = 0
for data in downloader:
    imgs, targets = data
    output = work(imgs)
    # print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1
