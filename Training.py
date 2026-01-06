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

class Training(ABC):
    @abstractmethod
    def __init__(self, dropout = 0) -> None:
        self.dropout_prob = dropout

    def _init_trainer(self):
        self.trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=1000,
            precision=32,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_l1",
                    save_top_k=1,
                    mode="min",
                    filename=(
                        f"{self.__class__.__name__}"
                        f"-drop-{self.dropout_prob:.1f}"
                        "-{epoch:03d}-{val_l1:.4f}"
                    ),
                    save_on_exception=True,
                ),
                EarlyStopping(monitor="val_l1", patience=50, mode="min"),
            ],
            log_every_n_steps=4,
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

    def _init_model(self):
        self.model = PanSharpenUnetppLightning(base_ch=32, lr=1e-4, dropout_prob=self.dropout_prob)

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

    for drp in [0]:
        training = PanSharpenUnetppLightningTraining(drp)
        trainingList.append(training)

    trainingList.listed_fit()
