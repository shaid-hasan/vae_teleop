'''
In this version chunk dataset is used without padding,
loader_var_chunk dataloader is used
fail, successfull, combined trial is visualized separately
'''
import collections
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
# from loader import *
from loader_var_chunk import *

wandb.login()
sweep_id = wandb.sweep(sweep_config, project="semi-adv-ae-sweep-t5-chunk")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


input_size = static_config['input_size']
num_layers = static_config['num_layers']
z_tsne_path = static_config['latent_z_tsne_path']
cat_tsne_path = static_config['cat_dim_tsne_path']
train_loss_path = static_config['train_loss_path']
saved_model_path = static_config['saved_model_path']
output_size = input_size
hidden_size = None
batch_size = None
chunk_size = None
seq_len = None
latent_dim = None
cat_dim = None
epochs = None
gen_lr = None
reg_lr = None
reg_lr_cat = None

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

    global hidden_size, batch_size, chunk_size, seq_len, latent_dim, cat_dim, epochs, gen_lr, reg_lr, reg_lr_cat

    with wandb.init(config=config):

         ## Config Extraction
        config = wandb.config
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
        

        ## Model Initialization
        Q = Q_net(input_size, hidden_size, latent_dim,cat_dim, num_layers).cuda()
        P = P_net(latent_dim+cat_dim, hidden_size, seq_len, output_size, num_layers).cuda()
        D_gauss = D_net_gauss(hidden_size,latent_dim).cuda()
        D_cat = D_net_cat(hidden_size,cat_dim).cuda()


        ## data loading
        train_loader, test_loader = load_dataloader(chunk_size, batch_size)

        # main training
        # Q_trained, recon_loss, disc_loss, gen_loss= train(Q, P, D_gauss, D_cat, train_loader)
        # torch.save(Q_trained.state_dict(), saved_model_path+f'semi_adv_ae_t5_chunk_{chunk_size}.pt')
        
        model_evaluation(test_loader,config)


def load_dataloader(chunk_size,batch_size):
    dataset = VR_input_Dataset()
    chunked_dataset = ChunkedDataset(original_dataset = VR_input_Dataset(), chunk_size=chunk_size)
    train_loader = DataLoader(chunked_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(chunked_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader


def train(Q, P, D_gauss, D_cat, train_loader):
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()
    optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
    optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
    optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
    optim_D_gauss = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)
    optim_D_cat = torch.optim.Adam(D_cat.parameters(), lr=reg_lr_cat)
    EPS = 1e-15

    Reconstruction_loss = []
    Discriminator_loss = []
    Generator_loss = []

    for epoch in range(epochs):
    
            total_recon_loss = 0
            total_D_loss = 0
            total_G_loss = 0
            num_batches = len(train_loader)

            for x, y, org_idx, chunk_idx in train_loader:

                X, Y = to_var(x), to_var(y)
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

    return Q, Reconstruction_loss, Discriminator_loss, Generator_loss


def model_evaluation(test_loader,config):

    model = Q_net(input_size, hidden_size, latent_dim,cat_dim, num_layers).cuda()
    model.load_state_dict(torch.load(saved_model_path+f'semi_adv_ae_t5_chunk_{chunk_size}.pt'))
    model.eval()

    all_z = []
    all_c = []
    all_y = []


    #####################################

    # last_n_chunk_idxs = collections.defaultdict(list) 
    # n = 10
    # for i, (x, y, org_idx, chunk_idx) in enumerate(test_loader):
    #     org_idx = org_idx.item()
    #     chunk_idx = chunk_idx.item()
    #     last_n_chunk_idxs[org_idx].append(chunk_idx)
    #     last_n_chunk_idxs[org_idx].sort(reverse=True)
    #     if len(last_n_chunk_idxs[org_idx]) > n:
    #         last_n_chunk_idxs[org_idx] = last_n_chunk_idxs[org_idx][:n]

    # for i, (x, y, org_idx, chunk_idx) in enumerate(test_loader):

    #     org_idx = org_idx.item()
    #     chunk_idx = chunk_idx.item()

    #     if chunk_idx in last_n_chunk_idxs[org_idx]:

    #         x, y = to_var(x), to_var(y)
    #         latent_z, latent_c = model(x.to(device))
    #         latent_z = latent_z.to('cpu').detach().numpy()
    #         latent_c = latent_c.to('cpu').detach().numpy()
    #         y = y.to('cpu').detach().numpy()
    #         all_z.append(latent_z)
    #         all_c.append(latent_c)
    #         all_y.append(y)


    # for i, (x, y, org_idx, chunk_idx) in enumerate(test_loader):
    #     x, y = to_var(x), to_var(y)
    #     latent_z, latent_c = model(x.to(device))
    #     latent_z = latent_z.to('cpu').detach().numpy()
    #     latent_c = latent_c.to('cpu').detach().numpy()
    #     y = y.to('cpu').detach().numpy()
    #     all_z.append(latent_z)
    #     all_c.append(latent_c)
    #     all_y.append(y)
#####################################

    first_n_chunk_idxs = collections.defaultdict(list)
    n = 50

    for i, (x, y, org_idx, chunk_idx) in enumerate(test_loader):
        org_idx = org_idx.item()
        chunk_idx = chunk_idx.item()
        
        first_n_chunk_idxs[org_idx].append(chunk_idx)
        first_n_chunk_idxs[org_idx] = sorted(first_n_chunk_idxs[org_idx][:n])

    for i, (x, y, org_idx, chunk_idx) in enumerate(test_loader):
        org_idx = org_idx.item()
        chunk_idx = chunk_idx.item()
        
        if chunk_idx in first_n_chunk_idxs[org_idx]:
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

    min_val_z = np.min(all_z_reduced, axis=0)
    max_val_z = np.max(all_z_reduced, axis=0)
    min_val_c = np.min(all_c_reduced, axis=0)
    max_val_c = np.max(all_c_reduced, axis=0)
    scaled_z_reduced = 20 * (all_z_reduced - min_val_z) / (max_val_z - min_val_z) - 10
    scaled_c_reduced = 20 * (all_c_reduced - min_val_c) / (max_val_c - min_val_c) - 10

    draw_plot(scaled_z_reduced, all_y, n, 'z')
    draw_plot(scaled_c_reduced, all_y, n, 'cat')

def draw_plot(all_latent, all_y, n, plot_type):

    plt.figure(figsize=(16, 6))

    # Subplot 1: i=0
    plt.subplot(1, 3, 1)
    plt.scatter(all_latent[all_y == 0, 0], all_latent[all_y == 0, 1], label="0", color='green')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"{plot_type}: Successful Trial (last {n} chunks)")
    plt.legend()

    # Subplot 2: i=1
    plt.subplot(1, 3, 2)
    plt.scatter(all_latent[all_y == 1, 0], all_latent[all_y == 1, 1], label="1", color='red')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"{plot_type}: Failed Trial (last {n} chunks)")
    plt.legend()

    # Subplot 3: i=0 and i=1
    plt.subplot(1, 3, 3)
    for i in range(2):
        color = 'red' if i == 1 else 'green'
        plt.scatter(all_latent[all_y == i, 0], all_latent[all_y == i, 1], label=str(i), color=color)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"{plot_type}: Successful+Failed Trial Combined (last {n} chunks)")
    plt.legend()

    plt.tight_layout()

    file_name = f"test_{plot_type}.png"
    file_path = file_name
    plt.savefig(file_path)
    im = plt.imread(file_path)
    wandb.log({f"{plot_type}-tsne": [wandb.Image(im, caption=f"{plot_type}_tsne")]})

wandb.agent(sweep_id, sweep_function) 