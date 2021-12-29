import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels): # aaa
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),  # stride=2 -> 반으로 줄어듦
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # |x| = (batch_size, in_channels, h, w)

        y = self.layers(x)
        # |y| = (batch_size, out_channels, h, w)

        return y


class ConvolutionalClassifier(nn.Module):

    def __init__(self, output_size):
        self.output_size = output_size

        super().__init__()

        self.blocks = nn.Sequential( # |x| = (n, 1, 28, 28)
            ConvolutionBlock(1, 32), # (n, 32, 14, 14)
            ConvolutionBlock(32, 64), # (n, 64, 7, 7) dd
            ConvolutionBlock(64, 128), # (n, 128, 4, 4) 공식대로 계산해보면 4가 됨
            ConvolutionBlock(128, 256), # (n, 256, 2, 2)
            ConvolutionBlock(256, 512), # (n, 512, 1, 1) 관습적으로 2배씩 늘려서 감
        )
        self.layers = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        assert x.dim() > 2
        # Flatten 하면 안됨, 2이상이면 죽여줌

        if x.dim() == 3:
            # |x| = (batch_size, h, w) (bs, 28, 28)
            x = x.view(-1, 1, x.size(-2), x.size(-1))
        # |x| = (batch_size, 1, h, w) (bs,1,28,28)

        z = self.blocks(x)
        # |z| = (batch_size, 512, 1, 1)

        y = self.layers(z.squeeze())
        # |y| = (batch_size, output_size)

        return y
