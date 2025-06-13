# perceiver_timeseries/datamodule.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class SlidingWindowDataset(Dataset):
    def __init__(self, csv_path, in_len, out_len, stride=1000):
        import numpy as np, pandas as pd, torch

        df = pd.read_csv(
            csv_path,
             usecols=list(range(1, 8)),# keep seven numeric columns
            dtype=np.float32,
            low_memory=False,
        )

        data = df.values            # ndarray float32, shape [T, 7]
        self.x, self.y = [], []
        for start in range(0, len(data) - in_len - out_len + 1, stride):
            end_in = start + in_len
            end_out = end_in + out_len
            self.x.append(data[start:end_in])
            self.y.append(data[end_in:end_out])
        self.x = torch.tensor(self.x)  # [N, in_len, 7]
        self.y = torch.tensor(self.y)  # [N, out_len, 7]


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"inputs": self.x[idx], "targets": self.y[idx]}

class CSVDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, test_path,
                 in_len=4096, out_len=5000, batch_size=8, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
    def setup(self, stage=None):
        hp = self.hparams
        self.train_ds = SlidingWindowDataset(hp.train_path, hp.in_len, hp.out_len)
        self.val_ds   = SlidingWindowDataset(hp.val_path,   hp.in_len, hp.out_len)
        self.test_ds  = SlidingWindowDataset(hp.test_path,  hp.in_len, hp.out_len)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers)
