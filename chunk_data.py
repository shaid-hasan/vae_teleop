import torch
import numpy as np


num_samples_class1 = 20
num_samples_class2 = 5
sequence_length = 1100
input_dim = 8


class1_data = [torch.randn(sequence_length, input_dim) for _ in range(num_samples_class1)]
class2_data = [torch.randn(sequence_length, input_dim) for _ in range(num_samples_class2)]

dataset = class1_data + class2_data


chunk_size = 10

chunked_data = []
labels = []

for i, data in enumerate(dataset):
    
    num_chunks = sequence_length // chunk_size
    
    
    for j in range(num_chunks):
        
        chunk = data[j*chunk_size:(j+1)*chunk_size, :]
        chunked_data.append(chunk)
        
        labels.append(0 if i < num_samples_class1 else 1)  # 0 for class 1, 1 for class 2


chunked_data = torch.stack(chunked_data)
labels = torch.tensor(labels)


shuffle_indices = torch.randperm(len(chunked_data))
chunked_data = chunked_data[shuffle_indices]
labels = labels[shuffle_indices]

print("Chunked Data Shape:", chunked_data.shape)
print("Labels Shape:", labels.shape)
