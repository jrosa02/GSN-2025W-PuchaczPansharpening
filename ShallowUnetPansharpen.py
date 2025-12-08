import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ShallowUNetPansharpen(nn.Module):
    """
    Shallow U-Net for pansharpening multispectral images using high-res RGB.

    Inputs:
      - rgb: tensor (B, 3, H, W)   -- high-resolution RGB
      - mul: tensor (B, C_ms, H_low, W_low) -- low-resolution multispectral (expected half res)

    Output:
      - pansharpened_mul: tensor (B, C_ms, H, W)
    """

    def __init__(self, n_ms_bands: int, base_channels: int = 32, use_bn: bool = True):
        """
        Args:
          n_ms_bands: number of multispectral bands to output (C_mul)
          base_channels: number of filters in the first conv (scales by 2 per level)
          use_bn: whether to use BatchNorm
        """
        super().__init__()
        self.n_ms_bands = n_ms_bands
        self.base = base_channels

        # encoder: input channels = rgb(3) + upsampled mul(n_ms_bands)
        in_ch = 3 + n_ms_bands

        # Level 1 (highest resolution)
        self.enc1 = nn.Sequential(
            conv_block(in_ch, self.base, use_bn=use_bn),
            conv_block(self.base, self.base, use_bn=use_bn),
        )
        # Level 2
        self.down1 = nn.Conv2d(self.base, self.base * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            conv_block(self.base * 2, self.base * 2, use_bn=use_bn),
            conv_block(self.base * 2, self.base * 2, use_bn=use_bn),
        )
        # Level 3 (bottleneck)
        self.down2 = nn.Conv2d(self.base * 2, self.base * 4, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Sequential(
            conv_block(self.base * 4, self.base * 4, use_bn=use_bn),
            conv_block(self.base * 4, self.base * 4, use_bn=use_bn),
        )

        # decoder
        self.up2 = nn.ConvTranspose2d(self.base * 4, self.base * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            conv_block(self.base * 4, self.base * 2, use_bn=use_bn),
            conv_block(self.base * 2, self.base * 2, use_bn=use_bn),
        )

        self.up1 = nn.ConvTranspose2d(self.base * 2, self.base, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            conv_block(self.base * 2, self.base, use_bn=use_bn),
            conv_block(self.base, self.base, use_bn=use_bn),
        )

        # final projection to multispectral bands
        self.head = nn.Conv2d(self.base, n_ms_bands, kernel_size=1)

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, rgb: torch.Tensor, mul: torch.Tensor) -> torch.Tensor:
        """
        Args:
          rgb: (B,3,H,W)
          mul: (B, C_ms, H_low, W_low) -- typically H_low = H/2, W_low = W/2

        Returns:
          out: (B, C_ms, H, W)
        """
        # sanity checks
        if rgb.dim() != 4 or mul.dim() != 4:
            raise ValueError("rgb and mul must be 4-D tensors (B,C,H,W)")

        # Upsample multspectral to rgb resolution (bilinear, align_corners=False)
        B, C_ms, H_low, W_low = mul.shape
        _, _, H, W = rgb.shape

        # If mul is already at same size, skip interpolation
        if (H_low, W_low) != (H, W):
            mul_up = F.interpolate(mul, size=(H, W), mode="bilinear", align_corners=False)
        else:
            mul_up = mul

        # concatenate along channel dimension
        x = torch.cat([rgb, mul_up], dim=1)  # (B, 3 + C_ms, H, W)

        # Encoder
        e1 = self.enc1(x)     # (B, base, H, W)
        d1 = self.down1(e1)   # (B, base*2, H/2, W/2)

        e2 = self.enc2(d1)    # (B, base*2, H/2, W/2)
        d2 = self.down2(e2)   # (B, base*4, H/4, W/4)

        # Bottleneck
        e3 = self.enc3(d2)    # (B, base*4, H/4, W/4)

        # Decoder
        u2 = self.up2(e3)     # (B, base*2, H/2, W/2)
        # concat skip from encoder level 2
        u2 = torch.cat([u2, e2], dim=1)  # (B, base*4, H/2, W/2)
        d3 = self.dec2(u2)    # (B, base*2, H/2, W/2)

        u1 = self.up1(d3)     # (B, base, H, W)
        # concat skip from encoder level 1
        u1 = torch.cat([u1, e1], dim=1)  # (B, base*2, H, W)
        d4 = self.dec1(u1)    # (B, base, H, W)

        out = self.head(d4)   # (B, C_ms, H, W)

        return out
    

