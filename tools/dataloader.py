import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np

class CustomDatasetRegression(Dataset):
    def __init__(self, path, dynamic, gas, step='trn', output_type='np', normalize=None):
        if step in ['trn', 'vld', 'tst']:
            self.fname = [fname for fname in os.listdir(os.path.join(path, dynamic)) if f'{step}_{gas}' in fname][0]
            self.global_fname = os.path.join(path, dynamic, self.fname)
            self.df = pd.read_csv(self.global_fname)
            self.df['idx'] = pd.factorize(self.df['Cycle'])[0]
        elif step == 'full':
            self.fnames = [fname for fname in os.listdir(os.path.join(path, dynamic)) 
                           if (f'trn_{gas}.csv') in fname or (f'vld_{gas}.csv') in fname or (f'tst_{gas}.csv') in fname
                            ]
            self.df = None
            for fname in self.fnames:
                global_fname = os.path.join(path, dynamic, fname)
                loaded_df = pd.read_csv(global_fname)
                self.df = loaded_df if self.df is None else pd.concat([self.df, loaded_df])
            self.df = self.df.sort_values(['Cycle', 'Tact'])
            self.df['idx'] = pd.factorize(self.df['Cycle'])[0]
        self.output_type = output_type
        self.SENSORS = ['R1','R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12']
        if normalize is not None:
            self.df = normalize(self.df, self.SENSORS)
    
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
    
    def get_all_data_by_step(self, sensor, target, step='trn'):
        assert self.output_type == 'np', "For using this method, self.output_type must be 'np'."
        df =self.get_df()
        grouped = df[df['Subset'] == step].groupby('Cycle')
        x_list = grouped[sensor].apply(list).tolist()
        y_list = grouped[target].apply(np.unique).tolist()
        return np.array(x_list), np.array(y_list)
    
    def get_df(self):
        return self.df.copy()
    

   
    
