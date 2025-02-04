import csv
import os
import shutil

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

# printing the field names
print('Field names are:' + ', '.join(field for field in fields))
# print(rows)

valid_data: list = []
for row in rows:
    if int(row[10]) == 0:
        valid_data.append(int(row[0]))

original_path: str = "./dataverse_files/cropped_thinwall_CSV_pyrometer/cropped_thinwall_CSV_pyrometer/"
final_path: str = "./dataset/correct/"

for file_number in valid_data:
    filename = f"Frame_{file_number}"

    source_path = os.path.join(original_path, filename)
    destination_path = os.path.join(final_path, filename + ".csv")

    # Move the file
    try:
        # Check if the source file exists before attempting to move
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"File moved to {destination_path}")
        else:
            print(f"Source file {filename} not found!")
    except Exception as e:
        print(f"Error: {e}")
