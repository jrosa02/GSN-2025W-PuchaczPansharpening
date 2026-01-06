from ConfigParser import ConfigParser
from SentinelData import SentinelCroppedDataset
from abc import ABC, abstractmethod
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
import torch.nn.functional as  F
import torch
from models.PansharpeningUnetppLightning import PanSharpenUnetppLightning

import cProfile
import traceback
import sys

class Training(ABC):
    @abstractmethod
    def __init__(self, dropout = 0) -> None:
        self.dropout_prob = dropout
        self.config = ConfigParser()

    def _init_trainer(self):
        self.trainer = pl.Trainer(
            accelerator= "cpu", #"gpu" if torch.cuda.is_available() else "cpu",
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
                    every_n_epochs=5,  # Save less often
                    save_on_train_epoch_end=False,  # Save after validation
                ),
                EarlyStopping(monitor="val_l1", 
                              patience=50, mode="min", 
                              check_on_train_epoch_end=False),
            ],
            log_every_n_steps=256
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
        self.dataloaders = self.dataset.produce_dataloaders(batch_size=self.config.get_training_batchsize()) #train, val, test

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

    torch.set_float32_matmul_precision('high')
    trainingList = TrainingList()

    for drp in [0, 0.5]:
        training = PanSharpenUnetppLightningTraining(drp)
        trainingList.append(training)

    trainingList.listed_fit()
