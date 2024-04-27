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
from torch.autograd import Variable
import wandb
import random
import time
import pprint
from config import *
from loader import *

wandb.login()
sweep_id = wandb.sweep(sweep_config, project="semi-adv-ae-sweep")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

#Encoder
class Q_net(nn.Module):  
    def __init__(self,input_size, hidden_size,z_dim, cat_dim, num_layers):
        super(Q_net, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size*2, num_layers=num_layers, batch_first=True)
        self.lin_gauss = nn.Linear(hidden_size*2, z_dim)
        self.lin_cat = nn.Linear(hidden_size*2, cat_dim)

    def forward(self, x):
        # print("\nEncoder:")
        out, hidden = self.gru(x)
        # print("out.shape(gru):",out.shape)
        out = out[:, -1, :]
        # print("out.shape(rearrange):",out.shape)
        # z = self.linear(out)
        z_gauss = self.lin_gauss(out)
        # print("z_gauss.shape:",z_gauss.shape)
        z_cat = self.lin_cat(out)
        # print("z_cat.shape:",z_cat.shape)

        return z_gauss, z_cat

# Decoder
class P_net(nn.Module):  
    def __init__(self, z_c_dim, hidden_size, seq_len, output_size, num_layers):
        super(P_net, self).__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size=z_c_dim, hidden_size=hidden_size*2, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, output_size)

    def forward(self, z, c):
        # print("\nDncoder:")
        latent_rep = torch.cat((z,c), 1)
        z_c = latent_rep.unsqueeze(1).repeat(1, self.seq_len, 1)
        # print("z_c.shape(unsqueeze):",z_c.shape)
        out, _ = self.gru(z_c)
        # print("out.shape(gru):",out.shape)
        x_hat = self.linear(out)
        
        return x_hat

# Discriminator
class D_net_gauss(nn.Module):  
    def __init__(self,hidden_size,z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))  

class D_net_cat(nn.Module):
    def __init__(self,hidden_size,cat_dim):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(cat_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))

def sample_categorical(batch_size, cat_dim):
    """
    Sample from a categorical distribution of size batch_size and # of classes
    return: torch.autograd.Variable with the sample
    """
    cat = np.random.randint(0, cat_dim, batch_size)
    cat = np.eye(cat_dim)[cat].astype('float32')
    cat = torch.from_numpy(cat)

    return Variable(cat)


def sweep_function(config=None):

    with wandb.init(config=config):

         ## Config Extraction
        config = wandb.config
        input_size = static_config['input_size']
        num_layers = static_config['num_layers']
        z_tsne_path = static_config['latent_z_tsne_path']
        cat_tsne_path = static_config['cat_dim_tsne_path']
        train_loss_path = static_config['train_loss_path']
        output_size = input_size

        hidden_size = config.hidden_size
        batch_size = config.batch_size

        chunk_size = config.chunk_size
        seq_len = chunk_size
        latent_dim = config.latent_dim_z
        cat_dim = config.cat_dim
        
        epochs = config.epochs
        gen_lr = config.gen_lr
        reg_lr = config.reg_lr
        reg_lr_cat = config.reg_lr_cat
        EPS = 1e-15

        ## Model Initialization
        Q = Q_net(input_size, hidden_size, latent_dim,cat_dim, num_layers).cuda()
        P = P_net(latent_dim+cat_dim, hidden_size, seq_len, output_size, num_layers).cuda()
        D_gauss = D_net_gauss(hidden_size,latent_dim).cuda()
        D_cat = D_net_cat(hidden_size,cat_dim).cuda()


        ## Trainer Initialization
        train_loader, test_loader = load_dataloader(chunk_size, batch_size)

        adversarial_loss = nn.BCELoss()
        reconstruction_loss = nn.MSELoss()

        optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
        optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
        optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
        optim_D_gauss = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)
        optim_D_cat = torch.optim.Adam(D_cat.parameters(), lr=reg_lr_cat)

        Reconstruction_loss = []
        Discriminator_loss = []
        Generator_loss = []

        ## Main training
        for epoch in range(epochs):
    
            total_recon_loss = 0
            total_D_loss = 0
            total_G_loss = 0
            num_batches = len(train_loader)

            for batch_idx, (X, Y) in enumerate(train_loader):

                X, Y = to_var(X), to_var(Y)
                # print("X.shape:",X.shape)
                # print("Y.shape:",Y.shape)

                P.zero_grad()
                Q.zero_grad()
                D_gauss.zero_grad()
                D_cat.zero_grad()
                Q.train()
                P.train()

                ################### Reconstruction Loss and Optimization of Q_enc and P ####################
                latent_z, latent_c = Q(X)  
                X_hat = P(latent_z, latent_c)
                recon_loss = reconstruction_loss(X_hat+EPS,X+EPS)      
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(P.parameters(), 10.0)
                torch.nn.utils.clip_grad_norm_(Q.parameters(), 10.0)
                optim_P.step()
                optim_Q_enc.step()
                total_recon_loss += recon_loss.item()
                # print(f"Epoch {epoch+1}: recon_loss: {recon_loss.item():.8f}")
                ################### ***************************************************** ####################

                
                ################### # Adversarial Loss and Optimization of D_gauss and D_cat ####################

                z_real_gauss = (torch.randn(X.size()[0], latent_dim) * 5.).cuda()
                z_real_cat = sample_categorical(X.size()[0], cat_dim).cuda()
                z_fake_gauss, z_fake_cat = Q(X)
                
                D_real_gauss = D_gauss(z_real_gauss)
                D_real_cat = D_cat(z_real_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)
                D_fake_cat = D_cat(z_fake_cat)

                # print("D_real_gauss.shape:",D_real_gauss.shape)
                # print("D_real_cat.shape:",D_real_cat.shape)
                # print("z_real_gauss.shape:",z_real_gauss.shape)
                # print("z_real_cat.shape:",z_real_cat.shape)
                

                D_loss_gauss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1- D_fake_gauss + EPS))
                D_loss_cat = -torch.mean(torch.log(D_real_cat + EPS) + torch.log(1- D_fake_cat + EPS))
                D_loss = D_loss_gauss + D_loss_cat

                Q.eval()
                D_loss.backward()
                optim_D_gauss.step()
                optim_D_cat.step()
                total_D_loss += D_loss.item()
                # print(f"Epoch {epoch+1}: D_loss: {D_loss.item():.8f}")
                
                ################### ***************************************************** ####################
                

                ################### # Adversarial Loss and Optimization of Q_gen ####################

                Q.train()
                z_fake_gauss, z_fake_cat = Q(X)
                D_fake_gauss = D_gauss(z_fake_gauss)
                D_fake_cat = D_cat(z_fake_cat)
                G_loss_gauss = -torch.mean(torch.log(D_fake_gauss)+EPS)
                G_loss_cat = -torch.mean(torch.log(D_fake_cat)+EPS)
                G_loss = G_loss_gauss + G_loss_cat
                G_loss.backward()
                torch.nn.utils.clip_grad_norm_(Q.parameters(), 10.0)
                optim_Q_gen.step()
                total_G_loss += G_loss.item()
                # print(f"Epoch {epoch+1}: G_loss: {G_loss.item():.8f}")
                ################### ***************************************************** ####################
                
            mean_recon_loss = total_recon_loss / num_batches
            mean_D_loss = total_D_loss / num_batches
            mean_G_loss = total_G_loss / num_batches

            Reconstruction_loss.append(recon_loss.item())
            Discriminator_loss.append(D_loss.item())  
            Generator_loss.append(G_loss.item())

            print(f"Epoch {epoch+1}: Mean recon_loss: {mean_recon_loss:.8f}, Mean D_loss: {mean_D_loss:.8f}, Mean G_loss: {mean_G_loss:.8f}")
            wandb.log({"epoch": epoch, "reconstruction_loss": mean_recon_loss, "Discriminator_loss": mean_D_loss, "Generator_loss": mean_G_loss})

        plot_losses_subplots_and_save(Reconstruction_loss, Discriminator_loss, Generator_loss, config, save_path=train_loss_path)
        plot_adver_semi_ae_latent(Q, test_loader,config, save_path_z = z_tsne_path, save_path_cat=cat_tsne_path)


def load_dataloader(chunk_size,batch_size):
    dataset = VR_input_Dataset()
    chunked_dataset = ChunkedDataset(original_dataset = VR_input_Dataset(), chunk_size=chunk_size)
    train_loader = DataLoader(chunked_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(chunked_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def plot_losses_subplots_and_save(Reconstruction_loss, Discriminator_loss, Generator_loss,config, save_path):
    epochs = range(len(Reconstruction_loss))

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(epochs, Reconstruction_loss, label='Reconstruction Loss', color='blue')
    axs[0].set_title('Reconstruction Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, Discriminator_loss, label='Discriminator Loss', color='orange')
    axs[1].set_title('Discriminator Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(epochs, Generator_loss, label='Generator Loss', color='green')
    axs[2].set_title('Generator Loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid(True)
    

    file_name = (f'loss_seq{config.chunk_size}_z{config.latent_dim_z}_c{config.cat_dim}_'
             f'e{config.epochs}_h{config.hidden_size}_b{config.batch_size}_'
             f'gen_lr{config.gen_lr}_lr_gauss{config.reg_lr}_lr_cat{config.reg_lr_cat}.png')

    file_path = save_path+file_name
    plt.savefig(file_path)


def plot_adver_semi_ae_latent(model, test_loader,config, save_path_z, save_path_cat):

    all_z = []
    all_c = []
    all_y = []
    model.eval()
    for i, (x, y) in enumerate(test_loader):
        x, y = to_var(x), to_var(y)
        latent_z, latent_c = model(x.to(device))
        latent_z = latent_z.to('cpu').detach().numpy()
        latent_c = latent_c.to('cpu').detach().numpy()
        y = y.to('cpu').detach().numpy()
        all_z.append(latent_z)
        all_c.append(latent_c)
        all_y.append(y)

    all_z = np.concatenate(all_z, axis=0)
    all_c = np.concatenate(all_c, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    tsne_z = TSNE(n_components=2, random_state=0, perplexity=10)
    tsne_c = TSNE(n_components=2, random_state=0, perplexity=10)
    all_z_reduced = tsne_z.fit_transform(all_z)
    all_c_reduced = tsne_c.fit_transform(all_c)

    all_y = all_y.reshape(-1)


    ## Plot for latent z
    plt.figure(figsize=(8, 8))
    for i in range(2):
        plt.scatter(all_z_reduced[all_y == i, 0], all_z_reduced[all_y == i, 1], label=str(i))

    plt.legend()
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    title = (f'z-t-SNE with seq:{config.chunk_size}, z:{config.latent_dim_z}, cat:{config.cat_dim},'
             f' ep:{config.epochs}, H:{config.hidden_size}, B:{config.batch_size},'
             f' lr:{config.gen_lr}, lr_g:{config.reg_lr}, lr_c:{config.reg_lr_cat}')

    plt.title(title)

    file_name = (f'z_tsne_seq{config.chunk_size}_z{config.latent_dim_z}_c{config.cat_dim}_'
             f'e{config.epochs}_h{config.hidden_size}_b{config.batch_size}_'
             f'gen_lr{config.gen_lr}_lr_g{config.reg_lr}_lr_c{config.reg_lr_cat}.png')

    file_path = save_path_z+file_name
    plt.savefig(file_path)
    img1 = plt.imread(file_path)
    wandb.log({"z_tsne": [wandb.Image(img1, caption="z_tsne")]})

    ## Plot for latent cat
    plt.figure(figsize=(8, 8))
    for i in range(2):
        plt.scatter(all_c_reduced[all_y == i, 0], all_c_reduced[all_y == i, 1], label=str(i))

    plt.legend()
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    title = (f'cat-t-SNE with seq:{config.chunk_size}, z:{config.latent_dim_z}, cat:{config.cat_dim},'
             f' ep:{config.epochs}, H:{config.hidden_size}, B:{config.batch_size},'
             f' gen_lr:{config.gen_lr}, lr_g:{config.reg_lr}, lr_c:{config.reg_lr_cat}')

    plt.title(title)

    file_name = (f'cat_tsne_seq{config.chunk_size}_z{config.latent_dim_z}_c{config.cat_dim}_'
             f'e{config.epochs}_h{config.hidden_size}_b{config.batch_size}_'
             f'gen_lr{config.gen_lr}_lr_g{config.reg_lr}_lr_c{config.reg_lr_cat}.png')
             
    file_path = save_path_cat+file_name
    plt.savefig(file_path)
    img2 = plt.imread(file_path)
    wandb.log({"cat_tsne": [wandb.Image(img2, caption="cat_tsne")]})


wandb.agent(sweep_id, sweep_function) 