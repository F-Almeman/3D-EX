import sys
import csv
import re
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict


# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('-c','--complete_dataset',help='Path of the complete unified dataset',required=True)
parser.add_argument('-i','--incomplete_dataset',help='Path of the incomplete unified dataset',required=True)
parser.add_argument('-o','--output_file',help='Path to the output unified dataset', default=False)

args = parser.parse_args()

cdf = pd.read_csv(args.complete_dataset,engine='python', na_values = [''], keep_default_na=False)
idf = pd.read_csv(args.incomplete_dataset,engine='python', na_values = [''], keep_default_na=False)

# We want unique (word, definition) tuples. So, we will group by ('WORD', 'DEFINITION') to get EXAMPLE and DATASET_NAME columns as lists
cdf_1 = cdf.groupby(['WORD', 'DEFINITION'], as_index=False)['EXAMPLE','DATASET_NAME'].agg(lambda x: list(x))

# Replace EXAMPLE, DATASET_NAME columns to a dictionary {example: dataset}
dic = []
for i, row in cdf_1.iterrows():
  t = tuple(zip(cdf_1.at[i,'EXAMPLE'], cdf_1.at[i,'DATASET_NAME']))
  
  md = dict()
  for k, v in t:
    md.setdefault(k, []).append(v)
  dic.append(md)

cdf_1["EXAMPL_DATASET"] = dic
cdf_1.drop(['EXAMPLE', 'DATASET_NAME'], inplace=True, axis=1)

# To combine examples of one dataset 
new_examples = []
new_datasets = []

for i, row in cdf_1.iterrows():
  d = cdf_1.at[i,'EXAMPL_DATASET']
  new = defaultdict(list)
  for k,v in d.items():
    new[repr(v)].append(k)
  new_examples.append(list(new.values()))
  new_datasets.append(list(new.keys()))

cdf_1['EXAMPLE'] = new_examples
cdf_1["DATASET_NAME"] = new_datasets

# Repeat the work for idf
idf_1= idf.groupby(['WORD', 'DEFINITION'], as_index=False)['DATASET_NAME'].agg(lambda x: list(x))

# Concat cdf and idf
cdf_1.drop(['EXAMPL_DATASET'], inplace=True, axis=1)
df = pd.concat([cdf_1, idf_1])

# Check for duplication between cdf and idf
df_duplicates = df[df.duplicated(['WORD', 'DEFINITION'], keep=False)]

# To solve the duplication problem: 
# 1) extract (delete) the duplication dataframe from the original one, so we will have two dataframes:
#   A) df_duplicates: the duplication dataframe 
#   B) df_1: the original dataframe - the duplication dataframe  
# 2) delete rows where EXAMPLE is NaN from df_duplicates (this will generate a dataframe with lengt half of the duplication dataframe)
# 3) merge the df_1 with the one resulting from step 2

# extract (delete) the duplication dataframe from the original one
df_1 = df[~df.duplicated(['WORD', 'DEFINITION'], keep=False)]

# delete rows where EXAMPLE is NaN
df_duplicates = df_duplicates.dropna(subset=['EXAMPLE'])

#concat df_1 with df_duplicates after removing rows where EXAMPLE is NaN
df_2 = pd.concat([df_1, df_duplicates])

# Lastly, check duplication (WORD, DEFINITION)
df_2[df_2.duplicated(['WORD', 'DEFINITION'], keep=False)]
df_2.sort_values("WORD")

df_2.to_csv(args.output_file, index = False, header=True)
