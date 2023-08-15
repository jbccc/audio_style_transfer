import torch
import torch.nn as nn
from typing import Any
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import numpy as np
    
class newVGG(nn.Module):
    def __init__(self):
        super(newVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 6)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        conv_1 = nn.Conv2d(1, 8, kernel_size = 5, stride = 2, padding = 2)
        conv_n = lambda n: nn.Conv2d(n, 2*n, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.relu = nn.ReLU()

        self.conv_block1 = nn.Sequential(
            conv_1,
            nn.BatchNorm2d(8),
            self.relu,
        )

        self.conv_block2 = nn.Sequential(
            conv_n(8),
            nn.BatchNorm2d(16),
            self.relu,
        )

        self.conv_block3 = nn.Sequential(
            conv_n(16),
            nn.BatchNorm2d(32),
            self.relu,
        )

        self.conv_block4 = nn.Sequential(
            conv_n(32),
            nn.BatchNorm2d(64),
            self.relu,
        )

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(64, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x= self.conv_block1(x)
        x= self.conv_block2(x)
        x= self.conv_block3(x)
        x= self.conv_block4(x)
        x= self.ap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    

class LitDiscriminator(pl.LightningModule):
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        if x.shape[0]<32:
            return None
        x = x.reshape(32, 1, 1025, 431)
        y_hat = self.discriminator(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train/loss", loss)
        self.log("train/acc", self.accuracy(y_hat, y))
        return loss
    
    def accuracy(self, y_hat, y):
        return (y_hat.argmax(1)==y).float().mean()
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class UlyNet(nn.Module):
    def __init__(self):
        super(UlyNet, self).__init__()

        N_CHANNELS = 1025
        N_FILTERS = 4096

        std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS)))
        kernel_data = np.random.randn(1, N_CHANNELS, N_FILTERS) * std
        
        # Convert the kernel data to a PyTorch tensor
        kernel = torch.tensor(kernel_data, dtype=torch.float32)
        
        # Define the convolutional layer
        self.conv = nn.Conv2d(N_CHANNELS, N_FILTERS, kernel_size=(N_CHANNELS, 1), bias=False)

        # Set the weight of the convolutional layer to the kernel data
        self.conv.weight = nn.Parameter(kernel)
    
    def forward(self, x):
        # Perform the convolution followed by ReLU activation
        print("here")
        net = nn.ReLU(self.conv(x))
        return net

if __name__=="__main__":
    x = torch.from_numpy(np.load("/data/conan/spectrogram_dataset/Maria - Piano/0.npy")).float().T.unsqueeze(0)
    model = Model()
    model.load_state_dict(torch.load("/data/conan/model/.pth"))
    print("net ok")
    y = model(x)
    print(y.shape)