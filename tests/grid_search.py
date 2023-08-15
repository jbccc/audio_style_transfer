import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from proj_sdsc.algorithm.gatys import Gatys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
n_it = 1000
content = torch.from_numpy(np.load("/data/conan/spectrogram_dataset/12Le Père Fouettard - Piano/2.npy")).reshape(1, 431, 1025)
style = torch.from_numpy(np.load("/data/conan/spectrogram_dataset/Fantasia - Sax/32.npy")).reshape(1, 431, 1025)

gatys = Gatys()
low_bound_alpha = 1e-3
high_bound_alpha = 1
low_bound_beta = 1e-2
high_bound_beta = 1

range_alpha = np.logspace(low_bound_alpha, high_bound_alpha, 5)
range_beta = np.logspace(low_bound_beta, high_bound_beta, 5)


map_val_loss = {
    (i, j):0
    for i in range_alpha
    for j in range_beta
}

for x in tqdm(map_val_loss):
    alpha, beta = x
    _, losses, _, _, _, _, _ = gatys.algorithm(content, style, n_it, "adam", n_it, alpha, beta)

    map_val_loss[x] = losses[-1]

val_loss_array = np.array([[map_val_loss[(i, j)] for j in range_beta] for i in range_alpha])
fig, ax = plt.subplots()
im = ax.imshow(val_loss_array, cmap='hot', origin='lower', extent=[low_bound_beta, high_bound_beta, low_bound_alpha, high_bound_alpha])
cbar = plt.colorbar(im, ax=ax)
ax.set_aspect('auto')
plt.xlabel('Beta')
plt.ylabel('Alpha')
plt.title('Heatmap of Log Loss')

plt.savefig("losses_grid.png")
