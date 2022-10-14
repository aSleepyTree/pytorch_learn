import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import ReLU, Sigmoid

dataset = torchvision.datasets.CIFAR10("/data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1, 2, 0, 3, 1], [0, 1, 2, 3, 1], [1, 2, 1, 0, 0], [5, 2, 3, 1, 1], [2, 1, 0, 1, 1]],
                     dtype=torch.float)

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.size())


class Work(nn.Module):
    def __init__(self):
        super(Work, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.maxpool1(input)
        output = self.sigmoid1(input)
        return output


work = Work()
# output = work(input)
# print(output)
writer = SummaryWriter("logs_01")
step = 0
for data in dataloader:

    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)

    output = work(imgs)
    writer.add_images("output", imgs, global_step=step)
    step += 1

writer.close()
