import os
import glob
import csv


'''
This code extracts all the data.pkl file paths from the VR interface data

Output:
data.pkl path list stored in a csv
'''

# Replace 'your_path' with the actual path you want to start from
folder_path = '/scratch/qmz9mg/vae/VR_Interface'

def find_pkl_files(path):
    pkl_files = glob.glob(os.path.join(path, '**/*.pkl'), recursive=True)
    return pkl_files

def write_paths_to_csv(file_path, paths):
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Paths"])  # Header
        for path in paths:
            csv_writer.writerow([path])


pkl_files = find_pkl_files(folder_path)

if pkl_files:
    csv_file_path = 'all_vr_pkl_path.csv'
    write_paths_to_csv(csv_file_path, pkl_files)
    print(f"List of .pkl files written to {csv_file_path}")
else:
    print("No .pkl files found.")




