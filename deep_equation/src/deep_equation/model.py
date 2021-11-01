import torch
import torch.nn as nn
from torchsummary import summary


def hot_encoding(operator):
    if operator == '+':
        return [1, 0, 0, 0]
    if operator == '-':
        return [0, 1, 0, 0]
    if operator == '*':
        return [0, 0, 1, 0]
    if operator == '/':
        return [0, 0, 0, 1]
    return [0, 0, 0, 0]


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

    def architect(self, image):
        image = self.conv1(image)
        image = self.tanh(image)
        image = self.avgpool(image)
        image = self.conv2(image)
        image = self.tanh(image)
        image = self.avgpool(image)
        image = self.conv3(image)
        image = self.tanh(image)

        image = image.reshape(image.shape[0], -1)
        image = self.linear1(image)
        image = self.tanh(image)
        image = self.linear2(image)

        return image

    # def forward(self, img1, img2):
    def forward(self, img):
        img = self.architect(img)
        # img1 = self.architect(img1)
        # img2 = self.architect(img2)
        # img = torch.cat((img1, img2), 1)
        return img


if __name__ == '__main__':
    model = LeNet()
    x = torch.randn(64, 1, 32, 32)
    # y = torch.randn(64, 1, 32, 32)
    # output = model(x, y)
    output = model(x)

    print(model)
    # summary(model, [(1, 32, 32), (1, 32, 32)])
    summary(model, (1, 32, 32))
    print("output.shape : ", output.shape)
