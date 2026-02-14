"""
EfficientNet-based U-Net implementation.

Uses EfficientNet as encoder for parameter-efficient feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResidualBlock(nn.Module):
    """Residual block for feature refinement"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class DecoderBlock(nn.Module):
    """
    Decoder block with transposed convolution and skip connections.
    
    Args:
        in_ch: Input channels from previous decoder layer
        skip_ch: Channels from skip connection
        out_ch: Output channels
        dropout: Dropout probability
    """
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.2):
        super().__init__()
        
        # Transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        
        # Convolution after concatenation with skip connection
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout),
            ResidualBlock(out_ch)
        )

    def forward(self, x, skip):
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder
        """
        # Upsample
        x = self.up(x)
        
        # Handle any size mismatches
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolutional processing
        return self.conv(x)


class EfficientNetB4Encoder(nn.Module):
    """
    EfficientNet-B4 encoder that extracts multi-scale features.
    
    Modified to accept grayscale (1-channel) input while using
    pretrained ImageNet weights.
    """
    def __init__(self, pretrained=True, in_channels=1):
        super().__init__()
        
        # Load EfficientNet-B4 with features_only mode
        # This returns intermediate features at different scales
        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)  # Select which stages to output
        )
        
        # Modify first conv layer for grayscale input if needed
        if in_channels != 3:
            # EfficientNet-B4 first conv: (3, 48, kernel_size=(3, 3), stride=(2, 2))
            original_conv = self.model.conv_stem
            self.model.conv_stem = nn.Conv2d(
                in_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Initialize new conv layer
            if pretrained and in_channels == 1:
                # Average the RGB weights for grayscale initialization
                with torch.no_grad():
                    self.model.conv_stem.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
        
        # Get channel information
        # EfficientNet-B4 channels at selected indices: [32, 56, 112, 160, 448]
        self.out_channels = [32, 56, 112, 160, 448]

    def forward(self, x):
        """
        Extract multi-scale features from input.
        
        Args:
            x: Input tensor (B, 1, H, W)
            
        Returns:
            List of feature tensors at different scales
        """
        features = self.model(x)
        return features


class EfficientUNet(nn.Module):
    """
    U-Net with EfficientNet-B4 encoder.
    
    This is the corrected version from the original notebook with proper
    channel alignment and architecture.
    
    Args:
        encoder_name: EfficientNet variant (e.g., 'efficientnet_b4')
        pretrained: Use ImageNet pretrained weights
        in_channels: Number of input channels (1 for grayscale)
        out_channels: Number of output channels (1 for binary segmentation)
        decoder_channels: List of decoder channel counts
        dropout: Dropout probability
    """
    def __init__(
        self,
        encoder_name='efficientnet_b4',
        pretrained=True,
        in_channels=1,
        out_channels=1,
        decoder_channels=[512, 256, 128, 64],
        dropout=0.2
    ):
        super().__init__()
        
        # Encoder
        self.encoder = EfficientNetB4Encoder(pretrained, in_channels)
        enc_channels = self.encoder.out_channels  # [32, 56, 112, 160, 448]
        
        # Bottleneck - processes the deepest encoder features
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[-1], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(decoder_channels[0])
        )
        
        # Decoder blocks with skip connections
        # Each decoder block takes features from the previous layer and a skip connection
        self.dec3 = DecoderBlock(
            decoder_channels[0],  # 512 from bottleneck
            enc_channels[-2],     # 160 from encoder stage 4
            decoder_channels[1],  # 256 output
            dropout
        )
        
        self.dec2 = DecoderBlock(
            decoder_channels[1],  # 256 from dec3
            enc_channels[-3],     # 112 from encoder stage 3
            decoder_channels[2],  # 128 output
            dropout
        )
        
        self.dec1 = DecoderBlock(
            decoder_channels[2],  # 128 from dec2
            enc_channels[-4],     # 56 from encoder stage 2
            decoder_channels[3],  # 64 output
            dropout
        )
        
        # Additional upsampling to reach original resolution
        # EfficientNet reduces spatial dimensions significantly
        self.up_final1 = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3], 2, 2),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.up_final2 = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3], 2, 2),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Final 1x1 convolution to produce output
        self.final = nn.Conv2d(decoder_channels[3], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, out_channels, H, W)
        """
        # Get multi-scale features from encoder
        skips = self.encoder(x)
        # skips[0]: 32 channels  (stage 1)
        # skips[1]: 56 channels  (stage 2)
        # skips[2]: 112 channels (stage 3)
        # skips[3]: 160 channels (stage 4)
        # (There's also stage 5 with 448 channels as the last feature)
        
        # Bottleneck on the deepest features
        x = self.bottleneck(skips[-1])  # Process 448-channel features
        
        # Decoder path with skip connections
        x = self.dec3(x, skips[-2])  # Use 160-channel skip
        x = self.dec2(x, skips[-3])  # Use 112-channel skip
        x = self.dec1(x, skips[-4])  # Use 56-channel skip (we don't use skips[0] with 32 channels)
        
        # Additional upsampling to reach input resolution
        x = self.up_final1(x)
        x = self.up_final2(x)
        
        # Final convolution
        logits = self.final(x)
        
        return logits

    def get_num_params(self):
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing EfficientUNet...")
    model = EfficientUNet(pretrained=False, in_channels=1, out_channels=1)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Verify output size matches input
    assert y.shape[2:] == x.shape[2:], "Output size should match input size"
    print("✓ All tests passed!")
