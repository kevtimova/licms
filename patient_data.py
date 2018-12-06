import pandas as pd
import os
import numpy as np
import torch

from torch.utils.data.dataset import Dataset

class PatientsDataset(Dataset):
    def __init__(self, args):
        self.before_path = os.path.join(args.datadir, "sepsis_before.csv")
        self.after_path = os.path.join(args.datadir, "sepsis_after.csv")
        self.before_data = pd.read_csv(self.before_path, header=None)
        self.after_data = pd.read_csv(self.after_path, header=None)
        self.args = args
        self.im_size = int(self.before_data.size/len(self.before_data))

    def __getitem__(self, index):
        x_before = np.array(self.before_data.iloc[index])
        x_after = np.array(self.after_data.iloc[index])
        x_before = torch.FloatTensor(x_before)
        x_after = torch.FloatTensor(x_after)
        return (x_before, x_after)

    def __len__(self):
        len_before = len(self.before_data)
        len_after = len(self.after_data)
        assert(len_before == len_after)
        return len_before