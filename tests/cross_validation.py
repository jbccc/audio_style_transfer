import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
torch.cuda.set_device(0)
from proj_sdsc.model.dataset import SpectrogramDataset, cross_validation
from proj_sdsc import config
from proj_sdsc.model.train import train_cv as train
import time

BATCH_SIZE = 512

dataset = SpectrogramDataset(config.dataset["spectrogram"],)


folds = cross_validation(dataset, 4)
list_cv = []
# train model on each fold for CV
for i, fold in enumerate(folds):

    print(f"Fold {i+1}")
    tic = time.time()
    train_set, val_set = fold
    fold_train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    fold_val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    print("Training...")
    CV_model = train(fold_train_dataloader, epochs=20,)
    print("Evaluating...")
    CV_model.eval()

    with torch.no_grad():
        total_count = len(val_set)
        p = np.array([])
        t = np.array([])
        for (val_data, val_labels) in fold_val_dataloader:
            s = val_data.shape[0]
            val_data = val_data.reshape(s, 1, 431, 1025)
            val_data = val_data.to(config.device)

            pred = CV_model(val_data).cpu().detach().numpy()
            target = val_labels.cpu().numpy()
            p = np.concatenate((p, np.argmax(pred, axis=1)))
            t = np.concatenate((t, np.argmax(target, axis=1)))

        precision, recall = precision_score(t, p, average="micro"), recall_score(t, p, average="micro")
        print(f"Precision: {precision}, Recall: {recall}")
        list_cv.append([precision, recall])
    
    print(f"Fold {i+1} took {(time.time()-tic)/60:.2f} minutes")

np.save("cv.npy", np.array(list_cv))