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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ANN_Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(ANN_Encoder, self).__init__()
        self.linear1 = nn.Linear(8800, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class ANN_Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(ANN_Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 8800)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1,1100, 8))

class ANN_Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(ANN_Autoencoder, self).__init__()
        self.encoder = ANN_Encoder(latent_dims)
        self.decoder = ANN_Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class GRU_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers):
        super(GRU_Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        out, hidden = self.gru(x)
        out = out[:, -1, :]
        z = self.linear(out)
        return z

class GRU_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, seq_len, output_size, num_layers):
        super(GRU_Decoder, self).__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.gru(z)
        x_hat = self.linear(out)
        return x_hat

class GRU_Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers, seq_len):
        super(GRU_Autoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = GRU_Encoder(input_size, hidden_size, latent_dim, num_layers)
        self.decoder = GRU_Decoder(latent_dim, hidden_size, seq_len, input_size, num_layers)

    def forward(self, x):
        z= self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class ANN_Encoder_vae(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(ANN_Encoder_vae, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)           
        z = mean + var*epsilon                          
        return z

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.LeakyReLU(self.FC_input(x))
        mean = self.FC_mean(x)
        log_var  = self.FC_var(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return z, mean, log_var
    
class ANN_Decoder_vae(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(ANN_Decoder_vae, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        z = self.LeakyReLU(self.FC_hidden(z))
        z = torch.sigmoid(self.FC_output(z))
        x_hat = z.reshape((-1,1100, 8))
        return x_hat

class ANN_vae(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, seq_len):
        super(ANN_vae, self).__init__()
        self.encoder = ANN_Encoder_vae(input_dim=input_size*seq_len, hidden_dim=hidden_size, latent_dim=latent_dim)
        self.decoder = ANN_Decoder_vae(latent_dim=latent_dim, hidden_dim = hidden_size, output_dim = input_size*seq_len)
        
    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

class GRU_Encoder_vae(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers):
        super(GRU_Encoder_vae, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.FC_mean  = nn.Linear(hidden_size, latent_dim)
        self.FC_var   = nn.Linear (hidden_size, latent_dim)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)           
        z = mean + var*epsilon                          
        return z

    def forward(self, x):
        out, hidden = self.gru(x)
        out = out[:, -1, :]
        mean = self.FC_mean(out)
        log_var  = self.FC_var(out)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return z, mean, log_var

class GRU_Decoder_vae(nn.Module):
    def __init__(self, latent_dim, hidden_size, seq_len, output_size, num_layers):

        super(GRU_Decoder_vae, self).__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.gru(z)
        x_hat = self.linear(out)
        return x_hat

class GRU_vae(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers, seq_len):
        super(GRU_vae, self).__init__()
        self.encoder = GRU_Encoder_vae(input_size, hidden_size, latent_dim, num_layers)
        self.decoder = GRU_Decoder_vae (latent_dim, hidden_size, seq_len, input_size, num_layers)
        
    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
