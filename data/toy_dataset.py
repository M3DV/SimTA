import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from environ import ENVIRON_PATH


class ToyDataset(Dataset):

    data_dir = ENVIRON_PATH.data_dir
    params_file = "params.pickle"
    t_file = "t.pickle"
    x_file = "x.pickle"
    y_file = "y.pickle"
    y_true_file = "y_true.pickle"
    idx_train_file = "idx_train.pickle"
    idx_val_file = "idx_val.pickle"

    def __init__(self, subset="train"):
        self.subset = subset
        self.params = pickle.load(open(os.path.join(self.data_dir,
            self.params_file), "rb"))
        self.t = pickle.load(open(os.path.join(self.data_dir,
            self.t_file), "rb"))
        self.x = pickle.load(open(os.path.join(self.data_dir,
            self.x_file), "rb"))
        self.y = pickle.load(open(os.path.join(self.data_dir,
            self.y_file), "rb"))
        self.y_true = pickle.load(open(os.path.join(self.data_dir,
            self.y_true_file), "rb"))
        
        if self.subset == "train":
            idx_train = pickle.load(open(os.path.join(self.data_dir,
                self.idx_train_file), "rb"))
            self.t = self.t[idx_train]
            self.x = self.x[idx_train]
            self.y = self.y[idx_train]
            self.y_true = self.y_true[idx_train]
        else:
            idx_val = pickle.load(open(os.path.join(self.data_dir,
                self.idx_val_file), "rb"))
            self.t = self.t[idx_val]
            self.x = self.x[idx_val]
            self.y = self.y[idx_val]
            self.y_true = self.y_true[idx_val]

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.t[idx], dtype=torch.float),\
            torch.tensor(self.x[idx], dtype=torch.float),\
            torch.tensor(self.y[idx], dtype=torch.float),\
            torch.tensor(self.y_true[idx], dtype=torch.float)
    
    @staticmethod
    def get_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size)
