"""
Standard U-Net implementation for lung segmentation.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical 
Image Segmentation", MICCAI 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers with BatchNorm and ReLU.
    Standard building block in U-Net.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # Use bilinear upsampling or transposed convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle input sizes that are not perfectly divisible
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to produce output"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture.
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        out_channels: Number of output channels (1 for binary segmentation)
        base_channels: Number of feature channels in first layer
        depth: Number of downsampling/upsampling blocks
        bilinear: Use bilinear upsampling instead of transposed convolutions
        dropout: Dropout probability (0 to disable)
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, 
                 depth=4, bilinear=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.bilinear = bilinear

        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.down_blocks.append(Down(ch, ch * 2))
            ch = ch * 2
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.bottleneck = DoubleConv(ch, ch * 2 // factor)
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(Up(ch, ch // 2 // factor, bilinear))
            ch = ch // 2
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        # Output layer
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        # Store skip connections
        skips = []
        
        # Initial conv
        x = self.inc(x)
        skips.append(x)
        
        # Encoder
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        skips = skips[::-1]  # Reverse for decoder
        for i, up in enumerate(self.up_blocks):
            x = up(x, skips[i + 1])
        
        # Apply dropout before final layer
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Output
        logits = self.outc(x)
        return logits

    def get_num_params(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = UNet(in_channels=1, out_channels=1, base_channels=64, depth=4)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
