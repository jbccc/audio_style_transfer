import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from proj_sdsc.model.discriminator import Model
from proj_sdsc.model.dataset import SpectrogramDataset, random_split, cross_validation
from proj_sdsc import config
from proj_sdsc.model.train import train_cv as train
from tqdm import tqdm
torch.cuda.set_device(1)
BATCH_SIZE = 600

# load model
model = Model() 
model.load_state_dict(torch.load(os.path.join(config.model["model_path"], "model_retrained.pth")))
model = model.to(config.device)
# load data
dataset = SpectrogramDataset(config.dataset["spectrogram"],)
train_set, val_set = random_split(dataset, 0.2)
total_train, total_val = len(train_set), len(val_set)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

model.eval()

# check for overfitting/underfitting
print('Checking for overfitting/underfitting')
with torch.no_grad():
    train_loss = 0
    print('begin training loss')
    predictions = np.array([])
    test_labels = np.array([])
    for (data, target) in tqdm(train_loader):
        s = data.shape[0]
        data = data.reshape(s, 1, 431, 1025)

        data = data.to(config.device)
        target = target.to(config.device)

        train_outputs = model(data)
        train_loss += nn.CrossEntropyLoss(reduction="sum")(train_outputs.squeeze(), target)
        
        predictions = np.append(predictions, np.argmax(train_outputs.cpu().numpy(), axis=1))
        test_labels = np.append(test_labels, np.argmax(target.cpu().numpy(), axis = 1))

    train_loss /= total_train
    print(f"Training Loss: {train_loss.item()}")

    val_loss = 0
    for (data, target) in tqdm(val_loader):
        s = data.shape[0]
        data = data.reshape(s, 1, 431, 1025)

        data = data.to(config.device)
        target = target.to(config.device)

        val_outputs = model(data)
        val_loss += nn.CrossEntropyLoss(reduction="sum")(val_outputs, target)

        predictions = np.append(predictions, np.argmax(val_outputs.cpu().numpy(), axis=1))
        test_labels = np.append(test_labels, np.argmax(target.cpu().numpy(), axis = 1))
    
    val_loss /= total_val
    print(f"Validation Loss: {val_loss.item()}")

plt.bar(['Training Loss', 'Validation Loss'], [train_loss.item(), val_loss.item()])
plt.xlabel('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.title('Loss Histogram - Overfitting check')
plt.savefig("loss_histogram.png")
plt.clf()

# precision, recall = precision_score(test_labels, predictions, average="micro"), recall_score(test_labels, predictions, average="micro")
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.savefig("precision_recall_curve.png")
# plt.clf()

# print(f1_score(test_labels, predictions, average="macro"))


conf_matrix = confusion_matrix(test_labels, predictions)
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
