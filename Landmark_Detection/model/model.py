import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import numpy as np
from torch.autograd import Variable


class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = nn.Sequential(
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
            ),
            nn.ReLU(),
        )
        return block

    def reducing_block(
        self, in_channels, out_channels, kernel_size=3, padding=1, stride=2
    ):
        block = nn.Sequential(
            # nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                padding=1,
                stride=stride,
            ),
            nn.ReLU()
            # nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_block(
        self, in_channels, mid_channel, out_channels, kernel_size=3, stride=2
    ):
        block = nn.Sequential(
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=mid_channel,
                padding=1,
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(mid_channel),
            # nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding = 'same'),
            # nn.ReLU(),
            # nn.BatchNorm2d(mid_channel),
            nn.ConvTranspose2d(
                in_channels=mid_channel,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1,
            )
            # nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,padding=1, stride = stride),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=mid_channel,
                padding=1,
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(mid_channel),g
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=mid_channel,
                padding=1,
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(mid_channel),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=out_channels,
                padding=1,
            ),
            nn.ReLU()
            # nn.BatchNorm2d(out_channels)
        )
        return block

    def __init__(self, in_channel=1, out_channel=23):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(
            in_channels=in_channel, out_channels=64
        )
        self.reduce1 = self.reducing_block(in_channels=64, out_channels=64)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.reduce2 = self.reducing_block(in_channels=128, out_channels=128)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.reduce3 = self.reducing_block(in_channels=256, out_channels=256)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(512),
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(512),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        # Decode

        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

    def crop_and_concat(self, upsampled, bypass):

        return cat((upsampled, bypass), 1)

    def forward(self, x):
        # x=x.double()
        # Encode
        encode_block1 = self.conv_encode1(x)
        reduce_block1 = self.reduce1(encode_block1)

        encode_block2 = self.conv_encode2(reduce_block1)
        reduce_block2 = self.reduce2(encode_block2)

        encode_block3 = self.conv_encode3(reduce_block2)
        reduce_block3 = self.reduce3(encode_block3)

        # Bottleneck
        bottleneck1 = self.bottleneck(reduce_block3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)
        return final_layer
