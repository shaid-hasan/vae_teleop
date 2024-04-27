
import os
import csv
import copy
import pickle
import math

task_id = 5

good_trial_path = f'/scratch/qmz9mg/vae/path_analysis/Task_{task_id}_good_trial.csv'
bad_trial_path = f'/scratch/qmz9mg/vae/path_analysis/Task_{task_id}_bad_trial.csv'
modified_good_pkl_path = f"/scratch/qmz9mg/vae/Interface_data_modified/VR/Task_{task_id}/successful_trial"
modified_bad_pkl_path = f"/scratch/qmz9mg/vae/Interface_data_modified/VR/Task_{task_id}/failed_trial"

with open(good_trial_path, 'r') as csv_file:
    good_pkl_paths = csv_file.read().splitlines()

with open(bad_trial_path, 'r') as csv_file:
    bad_pkl_paths = csv_file.read().splitlines()

pkl_paths = good_pkl_paths + bad_pkl_paths

def modified_pkl_generate(originial_pkl_paths, modified_data_path):

    for pkl_path in originial_pkl_paths:

        with open(pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        relative_path = pkl_path.split('/VR_Interface/', 1)[-1]
        actions_length = len(data['actions'])
        rounded_length = math.floor(actions_length / 10) * 10
        print(f"l:{relative_path}: {actions_length}: {rounded_length}")
        new_data = data['actions'][:rounded_length]

        parts = pkl_path.split('/')
        participant = parts[-5].split('_')[-1]
        task = parts[-4].split('_')[-1]
        interface = parts[-3].split('_')[-1]
        trial = parts[-2].split('_')[-1]

        modified_pkl_name = f"P_{participant}_T_{task}_I_{interface}_Tr_{trial}_modified.pkl"
        modified_pkl_path = os.path.join(modified_data_path, modified_pkl_name)

        with open(modified_pkl_path, 'wb') as modified_pkl_file:
            pickle.dump(new_data, modified_pkl_file)



modified_pkl_generate(bad_pkl_paths, modified_bad_pkl_path)
modified_pkl_generate(good_pkl_paths, modified_good_pkl_path)




