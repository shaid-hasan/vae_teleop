import pickle
import pandas as pd

file_path = 'VR_Interface/Participant_5/Task_1/Interaface_3/Trial_1/data.pkl'
observations_path = 'observations.csv'
actions_path = 'actions.csv'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

if isinstance(data, dict):
    keys = data.keys()
    print("Keys in the .pkl file:", keys)
else:
    print("The loaded data is not a dictionary, keys are not applicable for this type of data.")



if 'observations' in data:
    observations_df = pd.DataFrame(data['observations'])

    # print("DataFrame from 'observations':")
    # print(observations_df.head(3))

    observations_df.to_csv(observations_path, index=False)
    # print(f"\nObservations DataFrame saved to {observations_path}")
else:
    print("The key 'observations' does not exist in the loaded data.")



if 'actions' in data:
    actions_df = pd.DataFrame(data['actions'])

    # print("DataFrame from 'actions':")
    # print(actions_df.head(3))

    actions_df.to_csv(actions_path, index=False)
    # print(f"\nObservations DataFrame saved to {actions_path}")
else:
    print("The key 'actions' does not exist in the loaded data.")
