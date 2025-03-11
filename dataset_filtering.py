import csv
import os
import numpy as np
import pandas as pd


filename: str = "./dataverse_files/thermal-porosity-table.csv"

# initializing the titles and rows list
fields = []
rows = []

# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

directory = "./dataset/all_data/"

Y_values: list = [int(row[10]) for row in rows]
X_values = []

files = [f for f in os.listdir(directory)]

files.sort(key=lambda x: int(x.split('_')[1]))

for file in files:
    file_path = os.path.join(directory, file)

    try:
        df = pd.read_csv(file_path).iloc[:, 1:] # Drops the index column

        if df.shape != (200, 201):
            print(df.shape)
        X_values.append(df.to_numpy())
    except Exception as e:
        print(f"Error reading {file}: {e}")


X_values = np.vstack(X_values)


X_values = X_values / np.max(np.abs(X_values))

np.savez("./dataset/dataset.npz",
        X_values=X_values,
         Y_valies=Y_values)