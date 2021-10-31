import torch
import torch.nn as nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0
        )
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, layer):
        layer = self.conv1(layer)
        layer = self.tanh(layer)
        layer = self.avgpool(layer)
        layer = self.conv2(layer)
        layer = self.tanh(layer)
        layer = self.avgpool(layer)
        layer = self.conv3(layer)
        layer = self.tanh(layer)

        layer = layer.reshape(layer.shape[0], -1)
        layer = self.linear1(layer)
        layer = self.tanh(layer)
        layer = self.linear2(layer)
        return layer


if __name__ == '__main__':
    model = LeNet()
    x = torch.randn(64, 1, 32, 32)
    output = model(x)

    print(model)
    summary(model, (1, 32, 32))
    print("output.shape : ", output.shape)
