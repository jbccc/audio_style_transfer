import os.path
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.accelerators import cuda
from proj_sdsc.model.dataset import LitDataModule
from proj_sdsc import config
from proj_sdsc.model.discriminator import LitDiscriminator, Model, newVGG, UlyNet
import torch
from tqdm import tqdm
import time
import numpy as np

def train(model, data_module, epochs, log_interval, save_dir, save_model, save_wandb):

    if save_wandb:
        wandb_logger = WandbLogger(
            project="audio-style-transfer", log_model='all')

    checkpoint_callback_loss = ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        save_top_k=3,
        every_n_train_steps=log_interval
    )
    checkpoint_callback_acc = ModelCheckpoint(
        monitor="train/acc",
        mode="min",
        save_top_k=3,
        every_n_train_steps=log_interval
    )
    accelerator = cuda.CUDAAccelerator()
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        logger=wandb_logger if save_wandb else False,
        log_every_n_steps=log_interval,
        default_root_dir=save_dir,
        callbacks=[checkpoint_callback_loss, checkpoint_callback_acc],
    )

    trainer.fit(model, data_module)

    if save_model:
        with torch.no_grad():
            torch.save(model.discriminator.state_dict(), os.path.join(save_dir, "modelVGG.pth"))

    return model

def train_cv(fold_dataloader, epochs):
    model = Model()
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for _ in tqdm(range(epochs)):
        model.train()
        for (data, target) in fold_dataloader:
            s = data.shape[0]
            data = data.reshape(s, 1, 431, 1025)

            data = data.to(config.device)
            target = target.to(config.device)

            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()
    
    return model

def train_plot_loss_acc(model, train_dataloader, val_dataloader, n_epochs, len_train, len_val):

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    

    total_time = 0

    for i in range(n_epochs):
        tic = time.time()
        model.train()
        optimizer.zero_grad()

        total_loss = 0
        current_acc_count = []

        print(f"beginning train epoch {i}")
        for (data, target) in tqdm(train_dataloader):
            s = data.shape[0]
            data = data.reshape(s, 1, 431, 1025)

            data = data.to(config.device)
            target = target.to(config.device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
            current_acc_count.append((outputs.argmax(dim=1) == target.argmax(dim=1)).sum().item())
        
        total_loss /= len_train


        train_loss.append(total_loss)
        train_acc.append(sum(current_acc_count)/len_train)

        plt.plot(train_loss)
        plt.savefig("train_loss.png")
        plt.clf()
        print("loss graph updated")

        model.eval()

        loss = 0
        current_acc_count = []

        print(f"beginning eval epoch {i}")
        with torch.no_grad():
            for  (data, target) in tqdm(val_dataloader):
                s = data.shape[0]
                data = data.reshape(s, 1, 431, 1025)

                data = data.to(config.device)
                target = target.to(config.device)

                outputs = model(data)
                loss += criterion(outputs, target)

                current_acc_count.append((outputs.argmax(dim=1) == target.argmax(dim=1)).sum().item())

        loss /= len_val

        val_loss.append(loss.item())
        val_acc.append(sum(current_acc_count)/len_val)
        
        tac = time.time()
        this_time = tac-tic
        total_time += this_time
        print(f"epoch {i} finished in {this_time:.0f} seconds, total time elapsed: {total_time/60:.1f} minutes ({total_time/(i+1)/60:.1f} minutes per epoch)")
        print(f"estimated time remaining: {total_time*(n_epochs-i-1)/(i+1)/60:.0f} minutes")

    return train_loss, train_acc, val_loss, val_acc, model

if __name__ == "__main__":
    # model = Model()
    # model.load_state_dict(torch.load("/data/conan/model/model_60_ep.pth"))
    # N_SAMPLES = 430
    # N_CHANNELS = 1025
    # N_FILTERS = 4096
    model = UlyNet()
    # model = newVGG()
    lit_model = LitDiscriminator(model)
    pl_train_loader = LitDataModule(config.dataset["spectrogram"], batch_size=8)
    train(model=lit_model, data_module=pl_train_loader, epochs=20, log_interval=1,
          save_dir=config.model["model_path"], save_model=True, save_wandb=False)
