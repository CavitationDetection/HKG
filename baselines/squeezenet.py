import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x), inplace=True)
        return torch.cat([
            F.relu(self.expand1x1(x), inplace=True),
            F.relu(self.expand3x3(x), inplace=True)
        ], 1)

class SqueezeNet(nn.Module):
    def __init__(self, version='1_1', num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FireModule(96, 16, 64, 64),
                FireModule(128, 16, 64, 64),
                FireModule(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FireModule(256, 32, 128, 128),
                FireModule(256, 48, 192, 192),
                FireModule(384, 48, 192, 192),
                FireModule(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FireModule(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FireModule(64, 16, 64, 64),
                FireModule(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FireModule(128, 32, 128, 128),
                FireModule(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FireModule(256, 48, 192, 192),
                FireModule(384, 48, 192, 192),
                FireModule(384, 64, 256, 256),
                FireModule(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version")

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)


def SqueezeNet1_1(**kwargs):
    model = SqueezeNet(version = '1_1', num_classes = 6)
    return model

def SqueezeNet1_0(**kwargs):
    model = SqueezeNet(version = '1_0', num_classes = 6)
    return model

