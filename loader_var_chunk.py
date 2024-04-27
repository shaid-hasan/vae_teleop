import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


task_id = 5

class VR_input_Dataset(Dataset):
    def __init__(self):
        self.vr_input_data_path = f"/scratch/qmz9mg/vae/Interface_data_modified/VR/Task_{task_id}/"
        dir_list = glob.glob(self.vr_input_data_path + "*")
        # print(dir_list)
        self.data = []

        for class_path in dir_list:
            class_name = class_path.split("/")[-1]
            for pkl_path in glob.glob(class_path + "/*.pkl"):
                self.data.append([pkl_path, class_name])
        
        # print(self.data)
        self.class_map = {"successful_trial": 0, "failed_trial" : 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pkl_path, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        class_id = torch.tensor([class_id])

        with open(pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        actions_tensor = torch.tensor(np.array(data), dtype=torch.float32)

        return actions_tensor, class_id

class ChunkedDataset(Dataset):
    def __init__(self, original_dataset, chunk_size):
        self.original_dataset = original_dataset
        self.chunk_size = chunk_size
        
    def __len__(self):
        return sum(len(seq) // self.chunk_size for seq, _ in self.original_dataset)
    
    def __getitem__(self, idx):
        seq_idx = 0
        while idx >= 0:
            seq, label = self.original_dataset[seq_idx]
            seq_len = len(seq)
            num_chunks = seq_len // self.chunk_size
            
            if idx < num_chunks:
                chunk_idx = idx
                break
            
            idx -= num_chunks
            seq_idx += 1
            
        start = chunk_idx * self.chunk_size
        end = (chunk_idx+1) * self.chunk_size   
        chunk = seq[start:end]
        return chunk, label, seq_idx+1, chunk_idx+1

if __name__ == "__main__":

    dataset = VR_input_Dataset()

    # for i in range(len(dataset)):
    #     print("X.shape:",dataset[i][0].shape)
    #     print("y:",dataset[i][1])

    # seq, label = dataset[idx]
    # print(len(seq))
    # print(label)

    chunked_dataset = ChunkedDataset(original_dataset = VR_input_Dataset(), chunk_size=100)
    print("\n",len(chunked_dataset))
    for i in range(len(chunked_dataset)):
        # print("X_id:", i+1, "y:", chunked_dataset[i][1], "org_idx:", chunked_dataset[i][2], "chunk_idx:", chunked_dataset[i][3])
        print("org_idx:", chunked_dataset[i][2], "chunk_idx:", chunked_dataset[i][3])

    ################ Chunk Distribution #################

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

    # print("Number of chunks for each org_idx:")
    # for org_idx, count in org_idx_chunk_count.items():
    #     print(f"org_idx: {org_idx}, chunk_count: {count}")

    print(f"Minimum number of chunks for an org_idx: {min_chunks}")
    print(f"Maximum number of chunks for an org_idx: {max_chunks}")
