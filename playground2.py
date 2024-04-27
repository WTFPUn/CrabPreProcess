import torch
import torch.nn as nn

in_channels = 64
num_hiddens = 128

conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)

# sample input
x = torch.randn(1, 64, 300, 225)

print(conv(x).shape)