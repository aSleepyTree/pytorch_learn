import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet('./data_ImageNet',split='train', download=True, transform=torchvision.transforms.ToTensor())

vgg16_true = torchvision.models.vgg16(
    pretrained=True)
vgg16_false = torchvision.models.vgg16(
    pretrained=False)
# print(vgg16_false)
# print(vgg16_true)


train_data = torchvision.datasets.CIFAR10(
    './data',train=True, transform=torchvision.transforms.ToTensor(), download=True
)

# 加入一个线性层输入100输出10
# vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
# print(vgg16_true)

# 或者可以直接修改最后一个线性层使之输出10，以flase为例
# vgg16_false.classifier[6] = nn.Linear(4096, 10)

