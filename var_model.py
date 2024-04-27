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
# from loader import *
from loader_var_chunk import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class GRU_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers):
        super(GRU_Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, latent_dim)

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, hidden = self.gru(packed_x)
        # Unpack the sequence
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        out = out[:, -1, :]
        z = self.linear(out)
        return z

class GRU_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, seq_len, output_size, num_layers):
        super(GRU_Decoder, self).__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, z, lengths):
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        # Pack the padded sequence
        packed_z = rnn_utils.pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed_z)
        # Unpack the sequence
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        x_hat = self.linear(out)
        return x_hat

class GRU_Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers, seq_len):
        super(GRU_Autoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = GRU_Encoder(input_size, hidden_size, latent_dim, num_layers)
        self.decoder = GRU_Decoder(latent_dim, hidden_size, seq_len, input_size, num_layers)

    def forward(self, x, lengths):
        z = self.encoder(x, lengths)
        x_hat = self.decoder(z, lengths)
        return x_hat
