'''
In this version chunk dataset is used without padding,
loader_var_chunk dataloader is used
fail, successfull, combined trial is visualized separately
'''

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import collections
import wandb
import random
import time
import pprint
from config_ae import *
from loader_var_chunk import *
from models import *

wandb.login()
sweep_id = wandb.sweep(sweep_config, project="auto-encoder-sweep-v2-t5")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_size = static_config['input_size']
num_layers = static_config['num_layers']
num_layers = static_config['num_layers']
z_tsne_path = static_config['z_tsne_path']
train_loss_path = static_config['train_loss_path']
saved_model_path = static_config['saved_model_path']
hidden_size = None
chunk_size = None
latent_dim = None
batch_size = None
seq_len = None
epochs = None
learning_rate = None

def sweep_function(config=None):
    global hidden_size, chunk_size, latent_dim, batch_size, seq_len, epochs, learning_rate

    with wandb.init(config=config):
        config = wandb.config
        hidden_size = config.hidden_size
        chunk_size = config.chunk_size
        latent_dim = config.latent_dim
        batch_size = config.batch_size
        seq_len = chunk_size
        epochs = config.epochs
        learning_rate = config.learning_rate

        train_loader, test_loader = load_dataloader(chunk_size, batch_size)

        # model = GRU_Autoencoder(input_size, hidden_size, latent_dim, num_layers, seq_len).to(device)
        # trained_model, train_loss = train(model, train_loader, epochs, batch_size,learning_rate)
        # torch.save(trained_model.state_dict(), saved_model_path+'autoencoder_t5_chunk_5.pt')
        
        model_load_evaluation(test_loader)

def load_dataloader(chunk_size,batch_size):
    dataset = VR_input_Dataset()
    chunked_dataset = ChunkedDataset(original_dataset = VR_input_Dataset(), chunk_size=chunk_size)
    train_loader = DataLoader(chunked_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(chunked_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader

def train(model, train_loader, epochs, batch_size, learning_rate):
    
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.MSELoss()
    train_loss = []
    wandb.watch(model, criterion, log="all", log_freq=10)

    # dataiter = iter(train_loader)
    # x, y, org_id, chunk_id = next(dataiter)
    # print("x.shape:", x.shape, " y.shape:", y.shape, " org_idx:", org_id, " chunk_idx:", chunk_id)

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y, org_idx, chunk_idx in train_loader:
            x = x.to(device) 
            opt.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat,x)
            running_loss += loss.item()
            loss.backward()
            opt.step()
        
        train_loss.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
        wandb.log({"train_loss": running_loss/len(train_loader), "epoch": epoch+1})
    
    return model, train_loss

def model_load_evaluation(test_loader):

    model = GRU_Autoencoder(input_size, hidden_size, latent_dim, num_layers, seq_len).to(device)
    model.load_state_dict(torch.load(saved_model_path+'autoencoder_t5_chunk_5.pt'))
    model.eval()

    all_z = []
    all_y = []

    last_n_chunk_idxs = collections.defaultdict(list) 
    n = 8
    for i, (x, y, org_idx, chunk_idx) in enumerate(test_loader):
        org_idx = org_idx.item()
        chunk_idx = chunk_idx.item()
        last_n_chunk_idxs[org_idx].append(chunk_idx)
        last_n_chunk_idxs[org_idx].sort(reverse=True)
        if len(last_n_chunk_idxs[org_idx]) > n:
            last_n_chunk_idxs[org_idx] = last_n_chunk_idxs[org_idx][:n]

    for i, (x, y, org_idx, chunk_idx) in enumerate(test_loader):
        org_idx = org_idx.item()
        chunk_idx = chunk_idx.item()
        
        if chunk_idx in last_n_chunk_idxs[org_idx]:
            x, y = x.to(device), y.to(device)
            z = model.encoder(x)
            z = z.to('cpu').detach().numpy()
            y = y.to('cpu').detach().numpy()
            
            all_z.append(z)
            all_y.append(y)

    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    all_z_reduced = tsne.fit_transform(all_z)    
    all_y = all_y.reshape(-1)

    min_val = np.min(all_z_reduced, axis=0)
    max_val = np.max(all_z_reduced, axis=0)

    # Scale to the desired range (-10 to +10)
    scaled_z_reduced = 20 * (all_z_reduced - min_val) / (max_val - min_val) - 10
    draw_plot(scaled_z_reduced,all_z, all_y, n)

def draw_plot(scaled_z_reduced,all_z, all_y,n):

    plt.figure(figsize=(24, 8))

    # Subplot 1: i=0
    plt.subplot(1, 3, 1)
    plt.scatter(scaled_z_reduced[all_y == 0, 0], scaled_z_reduced[all_y == 0, 1], label="0", color='green')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"Successful Trial (last {n} chunks)")
    plt.legend()

    # Subplot 2: i=1
    plt.subplot(1, 3, 2)
    plt.scatter(scaled_z_reduced[all_y == 1, 0], scaled_z_reduced[all_y == 1, 1], label="1", color='red')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"Failed Trial (last {n} chunks)")
    plt.legend()

    # Subplot 3: i=0 and i=1
    plt.subplot(1, 3, 3)
    for i in range(2):
        color = 'red' if i == 1 else 'green'
        plt.scatter(scaled_z_reduced[all_y == i, 0], scaled_z_reduced[all_y == i, 1], label=str(i), color=color)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"Successful+Failed Trial Combined (last {n} chunks)")
    plt.legend()

    plt.tight_layout()

    file_name = "test.png"
    file_path = file_name
    plt.savefig(file_path)
    im = plt.imread(file_path)
    wandb.log({"z-tsne": [wandb.Image(im, caption="z_tsne")]})

wandb.agent(sweep_id, sweep_function) 