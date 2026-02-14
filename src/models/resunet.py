"""
ResNet-based U-Net implementation with pretrained encoder.

Uses ResNet as encoder backbone for better feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualBlock(nn.Module):
    """Residual block for decoder"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection"""
    def __init__(self, in_channels, skip_channels, out_channels, 
                 use_residual=True, dropout=0.2):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        # Convolutions after concatenation
        self.conv1 = nn.Conv2d(
            out_channels + skip_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        # Optional residual block
        self.residual = ResidualBlock(out_channels) if use_residual else None

    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolution
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Residual block
        if self.residual is not None:
            x = self.residual(x)
        
        return x


class ResNetEncoder(nn.Module):
    """ResNet encoder that extracts multi-scale features"""
    def __init__(self, encoder_name='resnet34', pretrained=True, in_channels=1):
        super().__init__()
        
        # Load pretrained ResNet
        if encoder_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        # Modify first conv for grayscale input if needed
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Extract encoder layers
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 (or 256 for resnet50)
        self.layer3 = resnet.layer3  # 256 (or 512)
        self.layer4 = resnet.layer4  # 512 (or 2048)

    def forward(self, x):
        """
        Extract features at multiple scales.
        Returns list of features from different stages.
        """
        features = []
        
        # Stage 0: Initial conv + pool
        x = self.conv1(x)
        features.append(x)  # 64 channels
        
        x = self.maxpool(x)
        
        # Stage 1-4: ResNet blocks
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features


class ResUNet(nn.Module):
    """
    U-Net with ResNet encoder.
    
    Args:
        encoder_name: ResNet variant ('resnet18', 'resnet34', 'resnet50')
        encoder_weights: 'imagenet' for pretrained or None
        in_channels: Number of input channels
        classes: Number of output classes
        decoder_channels: List of decoder channel counts
        use_residual: Use residual blocks in decoder
        dropout: Dropout probability
    """
    def __init__(
        self,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1,
        decoder_channels=[512, 256, 128, 64, 32],
        use_residual=True,
        dropout=0.2
    ):
        super().__init__()
        
        # Encoder
        pretrained = (encoder_weights == 'imagenet')
        self.encoder = ResNetEncoder(encoder_name, pretrained, in_channels)
        enc_channels = self.encoder.encoder_channels
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[-1], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            ResidualBlock(decoder_channels[0]) if use_residual else nn.Identity()
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels) - 1):
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=decoder_channels[i],
                    skip_channels=enc_channels[-(i+2)],
                    out_channels=decoder_channels[i+1],
                    use_residual=use_residual,
                    dropout=dropout
                )
            )
        
        # Final output
        self.final_conv = nn.Conv2d(decoder_channels[-1], classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, classes, H, W)
        """
        # Get encoder features
        encoder_features = self.encoder(x)
        
        # Bottleneck
        x = self.bottleneck(encoder_features[-1])
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = -(i + 2)
            x = decoder_block(x, encoder_features[skip_idx])
        
        # Final output
        x = self.final_conv(x)
        
        # Upsample to original size if needed
        if x.shape[2:] != encoder_features[0].shape[2:]:
            x = F.interpolate(
                x, size=encoder_features[0].shape[2:], 
                mode='bilinear', align_corners=True
            )
        
        return x

    def get_num_params(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = ResUNet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1
    )
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
