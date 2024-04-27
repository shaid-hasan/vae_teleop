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
from loader_var_chunk import *
# from verbose_models import *
from models import *
# from var_model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_size = 8
hidden_size = 64
num_layers = 2
latent_dim = 12
cat_dim = 12
batch_size = 10
chunk_size = 100
z_red_dims = latent_dim
seq_len = chunk_size

dataset = VR_input_Dataset()
# for i in range(len(dataset)):
#     print("X.shape:",dataset[i][0].shape, "y:",dataset[i][1])

# train_loader_org = DataLoader(dataset, batch_size=1, shuffle=True)
# dataiter_org = iter(train_loader_org)
# x, y = next(dataiter_org)
# print("x.shape:", x.shape, " y.value:", y)




chunked_dataset = ChunkedDataset(original_dataset = VR_input_Dataset(), chunk_size=5)

# print("\n",len(chunked_dataset))
# for i in range(len(chunked_dataset)):
#     print("X_id:", i+1, "y:", chunked_dataset[i][1], "org_idx:", chunked_dataset[i][2], "chunk_idx:", chunked_dataset[i][3])

train_loader = DataLoader(chunked_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(chunked_dataset, batch_size=1, shuffle=True)


# dataiter = iter(train_loader)
# x, y, org_id, chunk_id = next(dataiter)


# gru_ae = GRU_Autoencoder(input_size, hidden_size, latent_dim, num_layers, seq_len)
# print(gru_ae)
# x_hat = gru_ae(x)

#################################  min-max chunk count ###################

org_idx_chunk_count = {}
for i in range(len(chunked_dataset)):
        org_idx = chunked_dataset[i][2]
        chunk_idx = chunked_dataset[i][3]
        
        if org_idx in org_idx_chunk_count:
            org_idx_chunk_count[org_idx] += 1
        else:
            org_idx_chunk_count[org_idx] = 1

min_chunks = min(org_idx_chunk_count.values())
max_chunks = max(org_idx_chunk_count.values())

print("Number of chunks for each org_idx:")
for org_idx, count in org_idx_chunk_count.items():
    print(f"org_idx: {org_idx}, chunk_count: {count}")

print(f"Minimum number of chunks for an org_idx: {min_chunks}")
print(f"Maximum number of chunks for an org_idx: {max_chunks}")

#################################  *********************** ###################


#################################  last n chunk sort ###################

import collections
last_n_chunk_idxs = collections.defaultdict(list) 
n = 3
for i, (x, y, org_idx, chunk_idx) in enumerate(test_loader):

    org_idx = org_idx.item()
    chunk_idx = chunk_idx.item()
    
    last_n_chunk_idxs[org_idx].append(chunk_idx)
    last_n_chunk_idxs[org_idx].sort(reverse=True)
    if len(last_n_chunk_idxs[org_idx]) > n:
        last_n_chunk_idxs[org_idx] = last_n_chunk_idxs[org_idx][:n]

print(last_n_chunk_idxs)

#################################  *********************** ###################