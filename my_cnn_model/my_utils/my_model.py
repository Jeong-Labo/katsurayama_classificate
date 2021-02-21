import torch
from torch import nn
from torchsummary import summary


class MyModel(nn.Module):
    def __init__(self, input_ch=3, output_ch=2):
        super(MyModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


def print_summary(input_ch):
    net = MyModel(input_ch=input_ch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    net = net.to(device)
    summary(net, input_size=(1, 256, 256))


if __name__ == '__main__':
    print_summary(input_ch=1)
