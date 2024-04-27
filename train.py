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

from loader import *
from models import *
# from models import ANN_Autoencoder,GRU_Autoencoder, ANN_vae, GRU_vae
from mpl_toolkits.mplot3d import Axes3D

def loss_function(x, x_hat, mean, log_var):
    # reconstruct_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    reconstruct_loss = ((x - x_hat)**2).sum()
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    loss = reconstruct_loss + KLD
    return loss

def train(autoencoder, train_loader, epochs, batch_size, learning_rate):
    
    opt = torch.optim.Adam(autoencoder.parameters(),lr=learning_rate)
    reconstruction_loss = nn.MSELoss()
    train_loss = []

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device) 
            opt.zero_grad()
            x_hat = autoencoder(x)
            # loss = ((x - x_hat)**2).sum()
            loss = reconstruction_loss(x_hat,x)
            running_loss += loss.item()
            loss.backward()
            opt.step()

        train_loss.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    return autoencoder, train_loss

def train_vae(model, train_loader, epochs, batch_size, learning_rate):
    
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    train_loss = []

    for epoch in range(epochs):

        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss.append(total_loss/batch_size)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", total_loss/batch_size)

    return model, train_loss


def plot_train_loss(train_loss, save_path):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_latent_2D_AE(model, train_loader, save_path):

    all_z = []
    all_y = []

    for i, (x, y) in enumerate(train_loader):
        z = autoencoder.encoder(x.to(device))
        # z, _, _ = model.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        y = y.to('cpu').detach().numpy()
        all_z.append(z)
        all_y.append(y)

    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    all_z_reduced = tsne.fit_transform(all_z)
    
    all_y = all_y.reshape(-1)

    plt.figure(figsize=(8, 8))
    for i in range(2):
        plt.scatter(all_z_reduced[all_y == i, 0], all_z_reduced[all_y == i, 1], label=str(i))

    plt.legend()
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title("t-SNE visualization") 
    plt.show()
    plt.savefig(save_path)

def plot_latent_2D_VAE(model, train_loader, save_path):

    all_z = []
    all_y = []

    for i, (x, y) in enumerate(train_loader):
        # z = autoencoder.encoder(x.to(device))
        z, _, _ = model.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        y = y.to('cpu').detach().numpy()
        all_z.append(z)
        all_y.append(y)

    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    all_z_reduced = tsne.fit_transform(all_z)
    
    all_y = all_y.reshape(-1)

    plt.figure(figsize=(8, 8))
    for i in range(2):
        plt.scatter(all_z_reduced[all_y == i, 0], all_z_reduced[all_y == i, 1], label=str(i))

    plt.legend()
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title("t-SNE visualization") 
    plt.show()
    plt.savefig(save_path)

def plot_latent_3D(model, train_data_loader, save_path):

    all_z = []
    all_y = []

    for i, (x, y) in enumerate(train_data_loader):
        # z = autoencoder.encoder(x.to(device))
        z, _, _ = model.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        y = y.to('cpu').detach().numpy()
        all_z.append(z)
        all_y.append(y)

    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    tsne = TSNE(n_components=3, random_state=0, perplexity=10)
    all_z_reduced = tsne.fit_transform(all_z)
    
    all_y = all_y.reshape(-1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(2):
        ax.scatter(all_z_reduced[all_y == i, 0], all_z_reduced[all_y == i, 1], all_z_reduced[all_y == i, 2], label=str(i))

    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('t-SNE feature 1')
    ax.set_ylabel('t-SNE feature 2')
    ax.set_zlabel('t-SNE feature 3')
    ax.set_title("t-SNE visualization") 
    plt.show()
    plt.savefig('vr_tsne_3d_vae.png')

input_size = 8
hidden_size = 64
num_layers = 2
chunk_size = 50
latent_dim = 12
batch_size = 10
seq_len = chunk_size
epochs = 4
learning_rate = 1e-3
train_loss = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = VR_input_Dataset()
chunked_dataset = ChunkedDataset(original_dataset = VR_input_Dataset(), chunk_size=chunk_size)
data_loader = DataLoader(chunked_dataset, batch_size=10, shuffle=True)
# train_loader = DataLoader(dataset, batch_size, shuffle=True)
dataiter = iter(data_loader)
x, y = next(dataiter)
# print("x.shape:", x.shape)
# print("X(cuda):",x.is_cuda)


autoencoder = GRU_Autoencoder(input_size, hidden_size, latent_dim, num_layers, seq_len).to(device)
autoencoder, train_loss = train(autoencoder, data_loader, epochs, batch_size,learning_rate)
plot_train_loss(train_loss, save_path='/scratch/qmz9mg/vae/results/AE_gru_loss.png')
# data_loade_eval = DataLoader(chunked_dataset, batch_size=1, shuffle=True)
# plot_latent_2D_AE(autoencoder, data_loade_eval, save_path='/scratch/qmz9mg/vae/results/AE_gru_tsne.png')


# model = GRU_vae(input_size, hidden_size, latent_dim, num_layers, seq_len).to(device)
# trained_model, train_loss = train_vae(model, data_loader, epochs, batch_size,learning_rate)
# plot_train_loss(train_loss, save_path='/scratch/qmz9mg/vae/results/VAE_gru_loss.png')
# data_loade_eval = DataLoader(chunked_dataset, batch_size=1, shuffle=True)
# plot_latent_2D_VAE(trained_model, data_loade_eval, save_path='/scratch/qmz9mg/vae/results/VAE_gru_tsne.png')

