#Contains all code related to the code,
# including network architecture, layers, and loss functions.
# models.py


import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1 = self.conv_block(3, 64, 2)
        self.conv2 = self.conv_block(64, 128, 2)
        self.conv3 = self.conv_block(128, 256, 3)
        self.conv4 = self.conv_block(256, 512, 3)
        self.conv5 = self.conv_block(512, 512, 3)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)  # Output size 1 for binary classification

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def conv_block(self, in_channels, out_channels, num_conv):
        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x
