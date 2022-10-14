import torch
import torchvision.models

# 模型加载1
model1 = torch.load('vgg16_save1.pth')

# 模型加载2
vgg16 = torchvision.models.vgg16(pretrained=False)
model2 = torch.load("vgg16_save2.pth")
vgg16.load_state_dict(model2)
print(vgg16)