from ConfigParser import ConfigParser
from SentinelData import SentinelCroppedDataset
from abc import ABC, abstractmethod
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn.functional as  F
import torch
from models.PansharpeningUnetLightning import PanSharpenUnetLightning

def compute_stats(dataloader):
    rgb_sum = rgb_sq = ms_sum = ms_sq = 0
    n = 0

    for (rgb, ms_lr), _ in dataloader:
        b = rgb.size(0)
        rgb_sum += rgb.mean(dim=[0, 2, 3]) * b
        rgb_sq  += (rgb ** 2).mean(dim=[0, 2, 3]) * b

        ms_sum += ms_lr.mean(dim=[0, 2, 3]) * b
        ms_sq  += (ms_lr ** 2).mean(dim=[0, 2, 3]) * b
        n += b

    rgb_mean = rgb_sum / n
    rgb_std  = (rgb_sq / n - rgb_mean ** 2).sqrt()

    ms_mean = ms_sum / n
    ms_std  = (ms_sq / n - ms_mean ** 2).sqrt()

    return rgb_mean, rgb_std, ms_mean, ms_std

class Training(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _init_trainer(self):
        pass

    @abstractmethod
    def _init_dataset(self):
        pass
    
    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def fit(self):
        pass

class PanSharpenUnetLightningTraining(Training):
    def __init__(self) -> None:
        super().__init__()
        self._init_dataset()
        self._init_trainer()
        self._init_model()

    def _init_trainer(self):
        self.trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=200,
            precision=16,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_l1",
                    save_top_k=1,
                    mode="min",
                    filename="pansharpen-unet-{epoch:03d}-{val_l1:.4f}"
                ),
                EarlyStopping(
                    monitor="val_l1",
                    patience=20,
                    mode="min"
                )
            ],
            log_every_n_steps=4
        )
        
    def _init_dataset(self):
        self.dataset = SentinelCroppedDataset("./dataset_sentinel")
        self.dataloaders = self.dataset.produce_dataloaders() #train, val, test
        self.normalization_params = compute_stats(self.dataloaders[0]) # rgb_mean, rgb_std, ms_mean, ms_std

    def _init_model(self):
        rgb_mean, rgb_std, ms_mean, ms_std = self.normalization_params
        self.model = PanSharpenUnetLightning(rgb_mean, rgb_std, ms_mean, ms_std, base_ch=32, lr=1e-4)

    def fit(self):
        self.trainer.fit(self.model, self.dataloaders[0], self.dataloaders[1])

if __name__ == "__main__":
    training = PanSharpenUnetLightningTraining()
    training.fit()