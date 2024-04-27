import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


task_id = 3

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
        self.class_map = {"failed_trial" : 0, "successful_trial": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pkl_path, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        class_id = torch.tensor([class_id])

        with open(pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        actions_list = data['actions']
        actions_tensor = torch.tensor(np.array(actions_list), dtype=torch.float32)

        return actions_tensor, class_id


class ChunkedDataset(Dataset):
    def __init__(self, original_dataset, chunk_size):
        # self.org_seq_length = 1100          ## task-5: 1100
        self.org_seq_length = 1000          ## task-3: 1000
        self.original_dataset = original_dataset
        self.chunk_size = chunk_size
        self.total_num_chunks = len(self.original_dataset) * (self.org_seq_length // chunk_size)
        self.chunks_per_seq = (self.org_seq_length // self.chunk_size)

    def __len__(self):
        return self.total_num_chunks

    def __getitem__(self, idx):

        original_idx = idx // self.chunks_per_seq
        chunk_idx = idx % self.chunks_per_seq
        original_data, label = self.original_dataset[original_idx]
        chunked_data = original_data[chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size]

        return chunked_data, label, chunk_idx+1


if __name__ == "__main__":

    dataset = VR_input_Dataset()
    chunked_dataset = ChunkedDataset(original_dataset = VR_input_Dataset(), chunk_size=10)
    print("\n",len(chunked_dataset))
    for i in range(len(chunked_dataset)):
        print("X_id:", i+1, "y:", chunked_dataset[i][1], "chunk_idx:", chunked_dataset[i][2])
        