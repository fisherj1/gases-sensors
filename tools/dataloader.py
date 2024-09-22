import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np

class CustomDatasetRegression(Dataset):
    def __init__(self, path, dynamic, gas, step='trn', output_type='np'):
        self.fname = [fname for fname in os.listdir(os.path.join(path, dynamic)) if f'{step}_{gas}' in fname][0]
        self.global_fname = os.path.join(path, dynamic, self.fname)
        self.df = pd.read_csv(self.global_fname)
        self.df['idx'] = pd.factorize(self.df['Cycle'])[0]

        self.output_type = output_type
    
    def __len__(self):
        return self.df['idx'].max()+1
    
    def __getitem__(self, idx, sensor, target):
        x = self.df[self.df['idx'] == idx][sensor]
        y = self.df[self.df['idx'] == idx][target].unique()
        assert len(y) == 1

        if self.output_type == 'np':
            return np.array(x), np.array(y)
        return torch.tensor(x.values), torch.tensor(y)
    
    def get_all_data(self, sensor, target):
        assert self.output_type == 'np', "For using this method, self.output_type must be 'np'."
        x_list = []
        y_list = []
        for idx in range(self.__len__()):
            x, y = self.__getitem__(idx, sensor, target)
            x_list.append(np.expand_dims(x, 0))
            y_list.append(np.expand_dims(y, 0))
        return np.concatenate(x_list, 0), np.concatenate(y_list, 0)
