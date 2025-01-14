import torch
import torch.nn as nn
from torchsummary import summary
from pathlib import Path
from PIL import Image
from deep_equation.helper import img_to_tensor, hot_encoding, load_classes


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.num_classes = len(load_classes())
        self.conv1 = nn.Conv2d(
            in_channels=3,  # original LeNet-5 reads just one channel (grayscale)
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
        self.linear2 = nn.Linear(84, self.num_classes)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, image):
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


if __name__ == '__main__':
    model = LeNet()
    img1 = Image.open(Path("../../resources/digit_a.png"))
    img2 = Image.open(Path("../../resources/digit_b.png"))
    op = hot_encoding('+')
    t1 = img_to_tensor(img1)
    t2 = img_to_tensor(img2)
    # x = torch.randn(1, 3, 32, 32)
    x = torch.cat((t1, t2, op))
    x = x[None, :]
    output = model(x)

    print(model)
    # summary(model, [(1, 32, 32), (1, 32, 32)])
    summary(model, (3, 32, 32))
    print("output.shape : ", output.shape)
