from os import listdir
from os.path import isfile, join
import pandas as pd
import argparse


# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('-i','--datasets_folder',help='Path of the folder containing all datasets',required=True)
parser.add_argument('-o','--output_file',help='Output big csv file with all the datasets concatenated',required=True)
parser.add_argument('-a','--add',help='Add File_type column', default=False)


# Parse the argument
args = parser.parse_args()
  
# Read csv files
folder_path = args.datasets_folder
csv_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

# Create empty list
dataframes_list = []
 
# Append datasets into the list
for i in range(len(csv_files)):
	temp_df = pd.read_csv(folder_path+csv_files[i], na_values = [""], keep_default_na=False, engine='python')
	if temp_df.empty:
		continue
	if args.add == True:
		temp_df["FILE_NAME"] = csv_files[i].replace(".csv","")
	dataframes_list.append(temp_df)
    
# Concatenate dataframes
df = pd.concat(dataframes_list).fillna("N/A")
df = df.drop_duplicates()
  
df.to_csv (args.output_file, index = False, header=True)
