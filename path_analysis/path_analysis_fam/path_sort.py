import pandas as pd

# Read the CSV file
file_path = "/scratch/qmz9mg/vae/path_analysis/all_vr_pkl_path.csv"
df = pd.read_csv(file_path, header=None, names=["Path"])

# Extract Participant_ID, Task_ID, Interface_ID, and Trial_ID from the path
df[["Participant_ID", "Task_ID", "Interface_ID", "Trial_ID"]] = df["Path"].str.extract(r'Participant_(\d+)/Task_(\d+)/Interaface_(\d+)/Trial_(\d+)/data.pkl')

# Replace missing values (NaN) with zeros and convert to integers
df[["Participant_ID", "Task_ID", "Interface_ID", "Trial_ID"]] = df[["Participant_ID", "Task_ID", "Interface_ID", "Trial_ID"]].fillna(0).astype(int)

# Sort the dataframe based on the IDs
df_sorted = df.sort_values(by=["Participant_ID", "Task_ID", "Interface_ID", "Trial_ID"])

# Save the sorted dataframe to a new CSV file
output_file_path = "/scratch/qmz9mg/vae/path_analysis/sorted_path.csv"
df_sorted.to_csv(output_file_path, index=False)

print("Sorted paths saved to:", output_file_path)
