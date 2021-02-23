# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 08:14:03 2021

@author: natsl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### 1. Block double conv

class block_doubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
### 2. Encoder Block

class encoder_block(nn.Module):
	
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
            block_doubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)
    
### 3. Decoder Block

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = block_doubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

### 4. Block de sortie du Unet

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

### 5. Architecture Unet

class Unet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1, bilinear = True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = block_doubleConv(n_channels, 64)
        self.down1 = encoder_block(64,128)
        self.down2 = encoder_block(128,256)
        self.down3 = encoder_block(256,512) 
        self.down4 = encoder_block(512,1024 // 2) 
        
        self.up1 = decoder_block(1024, 512 //2)
        self.up2 = decoder_block(512, 256 // 2)
        self.up3 = decoder_block(256, 128 // 2)
        self.up4 = decoder_block(128, 64)
        self.outc = OutConv(64, 8)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits