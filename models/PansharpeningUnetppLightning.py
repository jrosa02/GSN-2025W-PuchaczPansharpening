import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchview import draw_graph
import torch
from lightning_utilities.core.rank_zero import rank_zero_warn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_prob=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        
        # Add dropout after first activation if probability > 0
        if dropout_prob > 0:
            layers.append(nn.Dropout2d(p=dropout_prob))
        
        layers.extend([
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ])
        
        # Add dropout after second activation if probability > 0
        if dropout_prob > 0:
            layers.append(nn.Dropout2d(p=dropout_prob))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, base_ch=32, dropout_prob=0.0):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch, dropout_prob)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, dropout_prob)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, dropout_prob)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.enc1(x)          # H, W
        f2 = self.enc2(self.pool(f1))  # H/2
        f3 = self.enc3(self.pool(f2))  # H/4
        return f1, f2, f3


class Decoder(nn.Module):
    def __init__(self, base_ch=32, out_ch=4, dropout_prob=0.0):
        super().__init__()
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, dropout_prob)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, dropout_prob)

        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, f1, f2, f3):
        x = self.up2(f3)
        x = self.dec2(torch.cat([x, f2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, f1], dim=1))

        return self.out_conv(x)


def normalize(x, mean, std):
    return (x - mean[None, :, None, None]) / std[None, :, None, None]


def denormalize(x, mean, std):
    return x * std[None, :, None, None] + mean[None, :, None, None]


class PanSharpenUnetppLightning(pl.LightningModule):
    def __init__(
            self,
            rgb_mean=None,
            rgb_std=None,
            ms_mean=None,
            ms_std=None,
            base_ch=32,
            lr=1e-4,
            dropout_prob=0.0,  # Add dropout probability parameter
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["rgb_mean", "rgb_std", "ms_mean", "ms_std"])

        # Defaults if loading from checkpoint
        if rgb_mean is None:
            rgb_mean = torch.zeros(3)
        if rgb_std is None:
            rgb_std = torch.ones(3)
        if ms_mean is None:
            ms_mean = torch.zeros(4)
        if ms_std is None:
            ms_std = torch.ones(4)

        self.register_buffer("rgb_mean", torch.as_tensor(rgb_mean))
        self.register_buffer("rgb_std",  torch.as_tensor(rgb_std))
        self.register_buffer("ms_mean",  torch.as_tensor(ms_mean))
        self.register_buffer("ms_std",   torch.as_tensor(ms_std))

        self.rgb_encoder = Encoder(3, base_ch, dropout_prob)
        self.ms_encoder  = Encoder(4, base_ch, dropout_prob)

        # Add dropout to fusion layers if desired
        self.fuse1 = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.fuse2 = nn.Conv2d(base_ch * 4, base_ch * 2, 1)
        self.fuse3 = nn.Conv2d(base_ch * 8, base_ch * 4, 1)
        
        # Optional: add dropout after fusion
        self.fuse_dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.decoder = Decoder(base_ch, out_ch=4, dropout_prob=dropout_prob)
        self.loss_fn = nn.L1Loss()

    def forward(self, rgb, ms_lr):
        # --- Normalize inputs ---
        rgb = normalize(rgb, self.rgb_mean, self.rgb_std)
        ms_lr = normalize(ms_lr, self.ms_mean, self.ms_std)

        # Upsample MS to RGB resolution
        ms_up = F.interpolate(ms_lr, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        rgb_f1, rgb_f2, rgb_f3 = self.rgb_encoder(rgb)
        ms_f1,  ms_f2,  ms_f3  = self.ms_encoder(ms_up)

        f1 = self.fuse_dropout(self.fuse1(torch.cat([rgb_f1, ms_f1], dim=1)))
        f2 = self.fuse_dropout(self.fuse2(torch.cat([rgb_f2, ms_f2], dim=1)))
        f3 = self.fuse_dropout(self.fuse3(torch.cat([rgb_f3, ms_f3], dim=1)))

        delta_ms = self.decoder(f1, f2, f3)

        # Residual prediction (still normalized space)
        ms_hr_norm = ms_up + delta_ms

        return ms_hr_norm

    def training_step(self, batch):
        (rgb, ms_lr), ms_hr = batch
        # Normalize GT MS
        ms_hr_norm = normalize(ms_hr, self.ms_mean, self.ms_std)
        pred_norm = self(rgb, ms_lr)
        loss = self.loss_fn(pred_norm, ms_hr_norm)
        self.log("train_l1", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        # During validation, dropout is automatically disabled
        (rgb, ms_lr), ms_hr = batch
        ms_hr_norm = normalize(ms_hr, self.ms_mean, self.ms_std)
        pred_norm = self(rgb, ms_lr)
        loss = self.loss_fn(pred_norm, ms_hr_norm)
        self.log("val_l1", loss, prog_bar=True)

    def predict_step(self, batch):
        # During prediction, dropout is automatically disabled
        (rgb, ms_lr), ms_hr = batch
        pred_norm = self(rgb, ms_lr)
        # Denormalize output for inference
        pred = denormalize(pred_norm, self.ms_mean, self.ms_std)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_l1",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
                "name": None,
            },
        }
    
if __name__ == "__main__":
    model = PanSharpenUnetppLightning()
    rgb = torch.randn(1, 3, 256, 256)
    ms  = torch.randn(1, 4, 128, 128)

    graph = draw_graph(
        model,
        input_data=(rgb, ms),
        expand_nested=True,
        depth=2,
        graph_name="PanSharpen UNet++",
        save_graph=True,
        filename="pansharpen_unet_clean"
    )

