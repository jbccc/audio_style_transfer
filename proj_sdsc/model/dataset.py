import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from proj_sdsc import config
from pytorch_lightning.core.datamodule import LightningDataModule
from torchvision import transforms
from typing import Union
from proj_sdsc import config

class SpectrogramDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.classes = ["Guitar", "Piano", "Clarinet", "Sax", "Trumpet", "Organ"]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = self.make_dataset()

    def make_dataset(self):
        '''
        architecture of the dataset:
        path
        ├── music 1 - intrument 1
            ├── 0.npy
            ├── 1.npy
            ...
        ├── music 1 - intrument 2
        ├── music 1 - intrument 3
        ...
        '''
        samples = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if not file.endswith(".npy"):
                    continue
                npy_path = os.path.join(root, file)
                label_idx = self.class_to_idx[root.split("/")[-1].split(" - ")[-1]]
                label = torch.zeros(len(self.classes))
                
                label[label_idx] = 1

                samples.append((npy_path, label))

            
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = torch.from_numpy(np.load(img_path))
        if self.transform:
            img = self.transform(img)
        return img, label
    

class LitDataModule(LightningDataModule):
    def __init__(self, spectrogram_path, batch_size):
        super().__init__()
        self.spectrogram_path = spectrogram_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


    def setup(self, stage: Union[str, None] = None):
        self.train_dataset = SpectrogramDataset(self.spectrogram_path, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)


# creates a dataset random split for training and validation
def random_split(dataset:Dataset, val_split=0.2):
    train_size = int((1-val_split)*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

# creates n folds for cross validation
def cross_validation(dataset:Dataset, n_folds=5):

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    indices = np.array_split(indices, n_folds)

    folds = []
    for i in range(n_folds):

        train_indices = np.concatenate([indices[j] for j in range(n_folds) if j!=i])
        val_indices = indices[i]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        folds.append((train_dataset, val_dataset))
        
    return folds

if __name__=="__main__":
    dataset = SpectrogramDataset(config.dataset["spectrogram"])
    print(dataset)
    dataloader = DataLoader(dataset, batch_size=155, shuffle=True)
    for data, label in dataloader:
        print(data.shape)
        print(label.shape)
        break
