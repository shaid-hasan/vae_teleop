import os
import csv
import copy
import pickle

good_trial_path = '/scratch/qmz9mg/vae/path_analysis/Task_5_good_trial.csv'
bad_trial_path = '/scratch/qmz9mg/vae/path_analysis/Task_5_bad_trial.csv'
modified_good_pkl_path = "/scratch/qmz9mg/vae/Interface_data_modified/VR/Task_5/successful_trial"
modified_bad_pkl_path = "/scratch/qmz9mg/vae/Interface_data_modified/VR/Task_5/failed_trial"

max_sequence_length = 0

with open(good_trial_path, 'r') as csv_file:
    good_pkl_paths = csv_file.read().splitlines()

with open(bad_trial_path, 'r') as csv_file:
    bad_pkl_paths = csv_file.read().splitlines()

pkl_paths = good_pkl_paths + bad_pkl_paths

def max_sequence_length_calculation(pkl_file_paths):
    
    max_actions_length = 0

    for pkl_path in pkl_file_paths:
        try:
            with open(pkl_path, 'rb') as pkl_file:
                data = pickle.load(pkl_file)

            if 'actions' in data:
                # relative_path = pkl_path.split('/vae/', 1)[-1]
                actions_length = len(data['actions'])
                # print(f"VR input sequence length in {relative_path}: {actions_length}")

                if actions_length > max_actions_length:
                    max_actions_length = actions_length

            else:
                print(f"'actions' key not found")

        except Exception as e:
            print(f"Error reading {pkl_path}: {e}")

    return max_actions_length

max_sequence_length = max_sequence_length_calculation(pkl_paths)
print(max_sequence_length)


def modified_pkl_generate(originial_pkl_paths, modified_data_path):

    global max_sequence_length

    for pkl_path in originial_pkl_paths:

        with open(pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        new_data = copy.deepcopy(data)
        first_action = new_data['actions'][0]
        while len(new_data['actions']) < max_sequence_length:
            new_data['actions'].append(first_action) 

        parts = pkl_path.split('/')
        participant = parts[-5].split('_')[-1]
        task = parts[-4].split('_')[-1]
        interface = parts[-3].split('_')[-1]
        trial = parts[-2].split('_')[-1]

        modified_pkl_name = f"P_{participant}_T_{task}_I_{interface}_Tr_{trial}_modified.pkl"
        modified_pkl_path = os.path.join(modified_data_path, modified_pkl_name)

        with open(modified_pkl_path, 'wb') as modified_pkl_file:
            pickle.dump(new_data, modified_pkl_file)

modified_pkl_generate(good_pkl_paths, modified_good_pkl_path)
modified_pkl_generate(bad_pkl_paths, modified_bad_pkl_path)