import torch
import torch.nn as nn
from torchvision.models import resnet50


class NullModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, scalars):
        ret = torch.zeros((image.shape[0], 67))
        ret[:, 0] = 1
        return ret.to(image.device)


class LaudeAndOstermann(nn.Module):
    def __init__(self, pu_size=(32, 32), num_classes=67):
        super().__init__()
        # fmt: off
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=96,
                                kernel_size=4, stride=1,
                                padding="same",padding_mode="replicate")
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(in_channels=96, out_channels=256,
                                kernel_size=5, stride=1,
                                padding="same", padding_mode="replicate")
        self.layer4 = nn.ReLU()
        self.layer5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Linear(in_features=256 * pu_size[0] // 2 * pu_size[0] // 2,
                                out_features=1024)
        self.layer7 = nn.ReLU()
        self.layer8 = nn.Linear(in_features=1024, out_features=67)
        # fmt: on

    def forward(self, image, scalars):
        pred = self.layer1(image)
        pred = self.layer2(pred)
        pred = self.layer3(pred)
        pred = self.layer4(pred)
        pred = self.layer5(pred)
        pred = pred.flatten(start_dim=1)
        pred = self.layer6(pred)
        pred = self.layer7(pred)
        pred = self.layer8(pred)
        return pred


class LaudeAndOstermannPlusScalars(nn.Module):
    def __init__(self, pu_size=(32, 32), num_scalars=11, num_classes=67):
        super().__init__()
        # fmt: off
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=96,
                                kernel_size=4, stride=1,
                                padding="same",padding_mode="replicate")
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(in_channels=96, out_channels=256,
                                kernel_size=5, stride=1,
                                padding="same", padding_mode="replicate")
        self.layer4 = nn.ReLU()
        self.layer5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Linear(in_features=256 * pu_size[0] // 2 * pu_size[0] // 2 + num_scalars,
                                out_features=1024)
        self.layer7 = nn.ReLU()
        self.layer8 = nn.Linear(in_features=1024, out_features=67)
        # fmt: on

    def forward(self, image, scalars):
        pred = self.layer1(image)
        pred = self.layer2(pred)
        pred = self.layer3(pred)
        pred = self.layer4(pred)
        pred = self.layer5(pred)
        pred = pred.flatten(start_dim=1)
        pred = torch.cat([pred, scalars], dim=1)
        pred = self.layer6(pred)
        pred = self.layer7(pred)
        pred = self.layer8(pred)
        return pred
