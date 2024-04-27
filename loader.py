import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class VR_input_Dataset(Dataset):

    def __init__(self):

        self.vr_input_data_path = "/scratch/qmz9mg/vae/Interface_data_modified/VR/Task_5/"
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
        self.original_dataset = original_dataset
        self.chunk_size = chunk_size
        self.num_chunks = len(self.original_dataset) * (1100 // chunk_size)

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):

        original_idx = idx // (1100 // self.chunk_size)
        chunk_idx = idx % (1100 // self.chunk_size)

        original_data, label = self.original_dataset[original_idx]

        chunked_data = original_data[chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size]

        return chunked_data, label


if __name__ == "__main__":

    dataset = VR_input_Dataset()
    print(len(dataset))
    print(type(dataset[0]))
    print("X.shape:",dataset[0][0].shape)
    print("y:",dataset[0][1])

    chunked_dataset = ChunkedDataset(original_dataset = VR_input_Dataset(), chunk_size=10)
    # Test the new dataset
    print("\n",len(chunked_dataset))
    print(type(chunked_dataset[100]))
    print("X.shape:", chunked_dataset[100][0].shape)
    print("y:", chunked_dataset[100][1])