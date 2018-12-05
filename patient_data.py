import pandas as pd
import os
import numpy as np

from torch.utils.data.dataset import Dataset

class PatientsDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.before_path = os.path.join(data_path, "sepsis_before.csv")
        self.after_path = os.path.join(data_path, "sepsis_after.csv")
        self.before_data = pd.read_csv(self.before_path)
        self.after_data = pd.read_csv(self.after_path)
        self.transforms = transforms

    def __getitem__(self, index):
        x_before = np.array(self.before_data.iloc[index])
        x_after = np.array(self.after_data.iloc[index])
        if self.transforms is not None:
            x_before = self.transforms(x_before)
            x_after = self.transforms(x_after)
        return (x_before, x_after)

    def __len__(self):
        len_before = len(self.before_data)
        len_after = len(self.after_data)
        assert(len_before == len_after)
        return len_before