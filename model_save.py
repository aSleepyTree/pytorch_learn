import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 模型保存1:模型结构+参数
torch.save(vgg16, "vgg16_save1.pth")

# 模型保存2:只有参数，配合模型使用（推荐）
torch.save(vgg16.state_dict(),'vgg16_save2.pth')
