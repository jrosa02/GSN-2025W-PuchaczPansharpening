from ConfigParser import ConfigParser
from SentinelData import SentinelCroppedDataset
from abc import ABC, abstractmethod
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn.functional as  F
import torch
from models.PansharpeningUnetppLightning import PanSharpenUnetppLightning
from math import sqrt

import torch

def compute_stats(dataloader, device="cpu"):
    # Use float64 accumulation to avoid precision loss
    rgb_sum = torch.zeros(3, dtype=torch.float64, device=device)
    rgb_sq  = torch.zeros(3, dtype=torch.float64, device=device)
    ms_sum  = torch.zeros(4, dtype=torch.float64, device=device)
    ms_sq   = torch.zeros(4, dtype=torch.float64, device=device)
    n_pixels = 0

    for (rgb, ms_lr), _ in dataloader:
        # Move batch to device and convert to float64 for stability
        rgb = rgb.to(device, dtype=torch.float64)
        ms_lr = ms_lr.to(device, dtype=torch.float64)

        b, c_rgb, h, w = rgb.shape
        _, c_ms, _, _ = ms_lr.shape

        # Number of pixels in batch
        batch_pixels = b * h * w
        n_pixels += batch_pixels

        # Sum over batch & spatial dimensions
        rgb_sum += rgb.sum(dim=[0, 2, 3])
        rgb_sq  += (rgb ** 2).sum(dim=[0, 2, 3])

        ms_sum += ms_lr.sum(dim=[0, 2, 3])
        ms_sq  += (ms_lr ** 2).sum(dim=[0, 2, 3])

    # Compute mean and std
    rgb_mean = (rgb_sum / n_pixels).float()
    rgb_std  = ((rgb_sq / n_pixels - (rgb_sum / n_pixels) ** 2).sqrt()).float()

    ms_mean = (ms_sum / n_pixels).float()
    ms_std  = ((ms_sq / n_pixels - (ms_sum / n_pixels) ** 2).sqrt()).float()

    return rgb_mean, rgb_std, ms_mean, ms_std


class Training(ABC):
    @abstractmethod
    def __init__(self, dropout = 0) -> None:
        self.dropout_prob = dropout

    def _init_trainer(self):
        self.trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=1000,
            precision=16,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_l1",
                    save_top_k=1,
                    mode="min",
                    filename="{self.__class__.__name__:s}-drop-{self.dropout_prob:.1f}-{epoch:03d}-{val_l1:.4f}",
                    save_on_exception=True
                ),
                EarlyStopping(
                    monitor="val_l1",
                    patience=50,
                    mode="min"
                )
            ],
            log_every_n_steps=4
        )

    @abstractmethod
    def _init_dataset(self):
        pass
    
    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def fit(self):
        pass

class PanSharpenUnetppLightningTraining(Training):
    def __init__(self, dropout = 0) -> None:
        super().__init__(dropout)
        self._init_dataset()
        self._init_trainer()
        self._init_model()

    def _init_dataset(self):
        self.dataset = SentinelCroppedDataset("./dataset_sentinel")
        self.dataloaders = self.dataset.produce_dataloaders() #train, val, test
        self.normalization_params = compute_stats(self.dataloaders[0]) # rgb_mean, rgb_std, ms_mean, ms_std

    def _init_model(self):
        rgb_mean, rgb_std, ms_mean, ms_std = self.normalization_params
        self.model = PanSharpenUnetppLightning(rgb_mean, rgb_std, ms_mean, ms_std, base_ch=32, lr=1e-4, dropout_prob=self.dropout_prob)

    def fit(self):
        self.trainer.fit(self.model, self.dataloaders[0], self.dataloaders[1])

class TrainingList:
    def __init__(self) -> None:
        self.training_list: list[Training] = []

    def append(self, training: Training):
        self.training_list.append(training)

    def listed_fit(self):
        for training in self.training_list:
            training.fit()

if __name__ == "__main__":

    trainingList = TrainingList()

    for drp in [0, 0.2, 0.5]:
        training = PanSharpenUnetppLightningTraining(drp)
        trainingList.append(training)

    trainingList.listed_fit()
