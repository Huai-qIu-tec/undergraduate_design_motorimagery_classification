import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.Conv2d(
                in_channels=1,  # input shape (B, 1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 64),  # filter size
                padding=(0, 32),
                bias=False
            ),  # output shape (B, 8, C, T)
            nn.BatchNorm2d(8)  # output shape (B, 8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (B, 8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(22, 1),  # filter size
                bias=False
            ),  # output shape (B, 16, 1, T)
            nn.BatchNorm2d(16),  # output shape (B, 16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # input shape (B, 16, 1, T//4)
                out_channels=32,  # num_filters
                kernel_size=(1, 16),  # filter size
                padding=(0, 8),
                bias=False
            ),  # output shape (B, 32, 1, T//4)
            nn.Conv2d(
                in_channels=32,  # input shape (B, 16, 1, T//4)
                out_channels=32,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (B, 32, 1, T//4)
            nn.BatchNorm2d(32),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 16)),  # output shape (16, 1, T//64)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear((32 * 8), classes_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
