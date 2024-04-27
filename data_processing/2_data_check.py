import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


pkl_path = "/scratch/qmz9mg/vae/Interface_data_modified/VR/Task_3/failed_trial/P_1_T_3_I_3_Tr_1_modified.pkl"

with open(pkl_path, 'rb') as pkl_file:
    data = pickle.load(pkl_file)


# print(type(data['actions']))
# print(len(data['actions']))
# print(type(data['actions'][0]))
print(len(data))
print(type(data))
print(type(data[-1]))