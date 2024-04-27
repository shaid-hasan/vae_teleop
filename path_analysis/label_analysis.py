import pandas as pd

df = pd.read_csv('/scratch/qmz9mg/vae/path_analysis/user_2_21_taskA.csv')

# Filter rows for Task_ID 1, 3, 5, and 7 with Interface 'VR'
task_ids_to_analyze = [1, 3, 5, 7]

for task_id in task_ids_to_analyze:
    task_rows = df[(df['Task_ID'] == task_id) & (df['Interface'] == 'VR')]

    # print(f"\nTask {task_id} with VR rows:")
    # print(task_rows)

    # Count the occurrences of each unique value in the 'Trial_number' column
    trial_counts = task_rows['Trial_number'].value_counts()

    print(f"\nTrial Counts for Task {task_id}:")
    print(trial_counts)
