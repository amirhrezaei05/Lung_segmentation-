"""
EfficientNet-based U-Net implementation.

Uses EfficientNet as encoder for parameter-efficient feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        identity = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + identity)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.2):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            ResidualBlock(out_ch)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)



class EfficientNetB4Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        self.model.conv_stem =nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    def forward(self, x):
        features = self.model(x)
        return features


model = EfficientNetB4Encoder()
print(model)

class EfficientUNet(nn.Module):
    def __init__(self):
        super().__init__()



        self.encoder = EfficientNetB4Encoder()

        enc_channels = [32, 56, 112, 160,272,448]

        self.bottleneck = nn.Sequential(
            nn.Conv2d(448, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(512)
        )

        self.dec4 = DecoderBlock(512, 160, 256)
        self.dec3 = DecoderBlock(256, 112, 128)
        self.dec2 = DecoderBlock(128, 56, 64)
        self.dec1 = DecoderBlock(64, 32, 32)


        self.final = nn.Conv2d(32, 1, kernel_size=1)


    def forward(self, x):
        skips = self.encoder(x)

        x = self.bottleneck(skips[-1])

        x = self.dec4(x, skips[-2])
        x = self.dec3(x, skips[-3])
        x = self.dec2(x, skips[-4])


        return self.final(x)

class EfficientUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = EfficientNetB4Encoder()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(448, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(512)
        )

        self.dec3 = DecoderBlock(512, 160, 256)
        self.dec2 = DecoderBlock(256, 56, 128)
        self.dec1 = DecoderBlock(128, 32, 64)
        self.up_final1 = nn.Sequential(
        nn.ConvTranspose2d(64, 64, 2, 2),
        nn.LeakyReLU(0.1, inplace=True)
        )

        self.up_final2 = nn.Sequential(
        nn.ConvTranspose2d(64, 64, 2, 2),
        nn.LeakyReLU(0.1, inplace=True)
        )



        self.final = nn.Conv2d(64, 1, kernel_size=1)



    def forward(self, x):
        skips = self.encoder(x)


        x = self.bottleneck(skips[-1])

        x = self.dec3(x, skips[-2])
        x = self.dec2(x, skips[-3])
        x = self.dec1(x, skips[-4])
        x = self.up_final1(x)
        x=self.up_final2(x)
        return self.final(x)

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
