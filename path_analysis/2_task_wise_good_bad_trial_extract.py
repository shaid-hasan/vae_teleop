import csv
from collections import Counter

csv_file_path = "/scratch/qmz9mg/vae/path_analysis/all_vr_pkl_path.csv"

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Skip the header
    paths = [row[0] for row in csv_reader]

task_id = 1

task_specific_path_list= [
    path for path in paths 
    if f'Task_{task_id}' in path and any(f'/Participant_{i}/' in path for i in range(2, 22))
]

participant_numbers = [int(path.split("/Participant_")[1].split("/")[0]) for path in task_specific_path_list]
participant_counts = Counter(participant_numbers)
participants_single_trial = [participant for participant, count in participant_counts.items() if count == 1]
participants_two_trials = [participant for participant, count in participant_counts.items() if count == 2]

print("Single Trial:",participants_single_trial)
print("Two Trial:",participants_two_trials)

good_trial_list = []
bad_trial_list = []

for participant_id in participants_two_trials:
    good_trial_list.append('/scratch/qmz9mg/vae/VR_Interface/Participant_{}/Task_{}/Interaface_3/Trial_2/data.pkl'.format(participant_id,task_id))
    bad_trial_list.append('/scratch/qmz9mg/vae/VR_Interface/Participant_{}/Task_{}/Interaface_3/Trial_1/data.pkl'.format(participant_id,task_id))

for participant_id in participants_single_trial:
    good_trial_list.append('/scratch/qmz9mg/vae/VR_Interface/Participant_{}/Task_{}/Interaface_3/Trial_1/data.pkl'.format(participant_id,task_id))


# # Create a CSV file for good_trial_list
# good_csv_file_path = f"/scratch/qmz9mg/vae/path_analysis/Task_{task_id}_good_trial.csv"
# with open(good_csv_file_path, 'w', newline='') as good_csv_file:
#     csv_writer = csv.writer(good_csv_file)
#     csv_writer.writerows([[path] for path in good_trial_list])

# # Create a CSV file for bad_trial_list
# bad_csv_file_path = f"/scratch/qmz9mg/vae/path_analysis/Task_{task_id}_bad_trial.csv"
# with open(bad_csv_file_path, 'w', newline='') as bad_csv_file:
#     csv_writer = csv.writer(bad_csv_file)
#     csv_writer.writerows([[path] for path in bad_trial_list])

# print("CSV files created successfully.")
