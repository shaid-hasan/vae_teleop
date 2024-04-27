import pandas as pd

# Read sorted paths CSV
df_sorted = pd.read_csv("/scratch/qmz9mg/vae/path_analysis/sorted_path.csv")

# Read user_2_21 CSV
df_user = pd.read_csv("/scratch/qmz9mg/vae/path_analysis/user_2_21.csv")

# Merge the two DataFrames based on common columns
merged_df = pd.merge(df_sorted, df_user, on=['Participant_ID', 'Task_ID', 'Interface_ID', 'Trial_ID'], how='left')

# Rename Interface Familiarity column
merged_df.rename(columns={'Interface_Familiarity': 'Familiarity'}, inplace=True)

# Reorder the columns if needed
# merged_df = merged_df[['Participant_ID', 'Task_ID', 'Trial_ID', 'Interface_ID', 'Familiarity', 'path']]

# Write the merged DataFrame back to CSV
merged_df.to_csv("/scratch/qmz9mg/vae/path_analysis/sorted_paths_with_familiarity.csv", index=False)

print("Merged CSV file created successfully.")

