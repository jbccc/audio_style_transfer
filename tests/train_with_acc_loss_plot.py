import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from proj_sdsc.model.discriminator import Model
from proj_sdsc.model.dataset import SpectrogramDataset, random_split

torch.cuda.set_device(1)
from proj_sdsc import config
from proj_sdsc.model.train import train_plot_loss_acc as train

BATCH_SIZE = 512
# load model
model = Model() 
model.to(config.device)
# load data
dataset = SpectrogramDataset(config.dataset["spectrogram"],)
train_set, val_set = random_split(dataset, 0.2)
len_train, len_val = len(train_set), len(val_set)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

model.train()

train_loss, train_acc, val_loss, val_acc, model = train(model, train_loader, val_loader, 100, len_train, len_val)

plt.plot(train_loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.title("Loss for train vs val set")
plt.legend()
plt.savefig("train_loss_fin_100.png")
plt.clf()

plt.plot(train_acc, label="train_acc")
plt.plot(val_acc, label="val_acc")
plt.title("Accuracy on train vs val set")
plt.legend()
plt.savefig("train_acc_fin_100.png")
plt.clf()

np.save("data_train_100.npy", np.array([train_loss, train_acc, val_loss, val_acc]))
torch.save(model.state_dict(), os.path.join(config.model["model_path"], "model_retrained_100.pth"))