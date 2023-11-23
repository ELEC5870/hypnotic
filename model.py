import torch
import torch.nn as nn
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50()
        self.reduce = nn.Linear(1000, 250)
        self.regress = nn.Linear(250 + 11, 67)

    def forward(self, image, scalars):
        output = self.resnet(image)
        output = nn.functional.relu(self.reduce(output))
        output = torch.cat([output, scalars], dim=1)
        output = self.regress(output)
        return output
