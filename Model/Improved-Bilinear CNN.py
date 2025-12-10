import torch
import torch.nn as nn
from torchvision.models import resnet18


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(resnet18().conv1,
                                      resnet18().relu,
                                      resnet18().maxpool,
                                      resnet18().layer2,
                                      resnet18().layer3,
                                      resnet18().layer4)
        self.classifiers = nn.Sequential(nn.Linear(512 ** 2, 14))

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        feature_size = x.size(2) * x.size(3)
        x = x.view(batch_size, 512, feature_size)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = self.classifiers(x)
        return x
