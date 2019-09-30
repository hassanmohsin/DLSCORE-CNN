import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pdbmap = json.load(open(os.path.join(data_dir, 'pdbmap.json')))
        self.labels = json.load(open(os.path.join(data_dir, 'labels.json')))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        samples = torch.from_numpy(np.load(os.path.join(self.data_dir, 'id_'+str(idx)+'.npy')).astype(np.float32))#[np.newaxis, :] 
        # TODO: Check if the following permuation preserves the data
        samples = samples.permute(3, 0, 1, 2) # Get the channels at the beginning
        targets = torch.from_numpy(np.array(self.labels['id_'+str(idx)]).astype(np.float32))
        targets = targets.view(-1)

        return samples, targets

    def __len__(self):
        return len(self.labels)

