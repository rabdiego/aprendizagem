import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file : str) -> None:
        self.data = torch.from_numpy(pd.read_csv(csv_file, header=None).values).to(torch.float32)
    
    
    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, index) -> torch.tensor:
        attributes = self.data[index][:-1]
        labels = self.data[index][-1]
        return attributes, labels
    
    
    def normalize(self) -> None:
        self.data = StandardScaler().fit_transform(self.data.numpy())
        self.data = torch.from_numpy(self.data).to(torch.float32)

