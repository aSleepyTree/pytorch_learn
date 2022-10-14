import torch
from torch import nn


class Work(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, got):
        output = got + 1
        return output


work = Work()
x = torch.tensor(1.0)
output = work(x)
print(output)