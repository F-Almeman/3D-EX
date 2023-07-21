import pandas as pd
import argparse
import os
import sys
import csv
import re
import numpy as np

# Words splits
import warnings
import random
warnings.filterwarnings('ignore')

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-i','--dataset_file',help='Unified dataset',required=True)
  parser.add_argument('-o','--output_path',help='Path to output files (train/test/val)',required=True)

  args = parser.parse_args()

  # This block is added to be able to read this big file
  maxInt = sys.maxsize
  while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
        
  # Read the csv file
  unified_dataset = pd.read_csv(args.dataset_file, engine='python', na_values = [''], keep_default_na=False)

  # Random splits
  random_train, random_valid, random_test = \
              np.split(unified_dataset.sample(frac=1, random_state=42), 
                       [int(.6*len(unified_dataset)), int(.8*len(unified_dataset))])

  
  random_train.to_csv(os.path.join(args.output_path, "random_train.csv"), index = False, header=True)
  random_valid.to_csv(os.path.join(args.output_path, "random_valid.csv"), index = False, header=True)
  random_test.to_csv(os.path.join(args.output_path, "random_test.csv"), index = False, header=True)

  # Lexical splits
  train_words = set()
  valid_words = set()
  test_words = set()

  # WordNet
  WordNet = unified_dataset[unified_dataset.DATASET_NAME.str.contains("WordNet")]
  WordNet_words = WordNet['WORD'].unique().tolist()

  WordNet_train_set = pd.DataFrame()
  WordNet_valid_set = pd.DataFrame()
  WordNet_test_set = pd.DataFrame()

  WordNet_train_set_complete=False
  WordNet_valid_set_complete=False

  for w in WordNet_words:
    word_subset = WordNet[WordNet["WORD"] == w]

    if not WordNet_train_set_complete:
      WordNet_train_set = WordNet_train_set.append(word_subset)
      train_words.add(w)
    
      if len(WordNet_train_set) > len(WordNet)*0.6:
        WordNet_train_set_complete = True
      
    elif WordNet_train_set_complete and not WordNet_valid_set_complete:
      WordNet_valid_set = WordNet_valid_set.append(word_subset)
      valid_words.add(w)

      if len(WordNet_valid_set) > len(WordNet)*0.2:
        WordNet_valid_set_complete = True

    elif WordNet_train_set_complete and WordNet_valid_set_complete:
      WordNet_test_set = WordNet_test_set.append(word_subset)
      test_words.add(w)

  
  # Wiktionary
  Wiktionary = unified_dataset[unified_dataset.DATASET_NAME.str.contains("Wiktionary")]
  Wiktionary_words = Wiktionary['WORD'].unique().tolist()

  Wiktionary_train_set = pd.DataFrame()
  Wiktionary_valid_set = pd.DataFrame()
  Wiktionary_test_set = pd.DataFrame()

  Wiktionary_train_set_complete=False
  Wiktionary_valid_set_complete=False

  for w in Wiktionary_words:
    word_subset = Wiktionary[Wiktionary["WORD"] == w]

    if (not Wiktionary_train_set_complete and w not in valid_words and w not in test_words) or (w in train_words):
      Wiktionary_train_set = Wiktionary_train_set.append(word_subset)
      train_words.add(w)
    
      if len(Wiktionary_train_set) > len(Wiktionary)*0.6:
        Wiktionary_train_set_complete = True
      
    elif (Wiktionary_train_set_complete and not Wiktionary_valid_set_complete and w not in test_words) or (w in valid_words):
     Wiktionary_valid_set = Wiktionary_valid_set.append(word_subset)
     valid_words.add(w)
   
     if len(Wiktionary_valid_set) > len(Wiktionary)*0.2:
        Wiktionary_valid_set_complete = True

    elif (Wiktionary_train_set_complete and Wiktionary_valid_set_complete) or (w in test_words):
      Wiktionary_test_set = Wiktionary_test_set.append(word_subset)
      test_words.add(w)

  
  # Urban
  Urban = unified_dataset[unified_dataset.DATASET_NAME.str.contains("Urban")]
  Urban_words = Urban['WORD'].unique().tolist()

  Urban_train_set = pd.DataFrame()
  Urban_valid_set = pd.DataFrame()
  Urban_test_set = pd.DataFrame()

  Urban_train_set_complete=False
  Urban_valid_set_complete=False

  for w in Urban_words:
    word_subset = Urban[Urban["WORD"] == w]

    if (not Urban_train_set_complete and w not in valid_words and w not in test_words) or (w in train_words):
      Urban_train_set = Urban_train_set.append(word_subset)
      train_words.add(w)
    
      if len(Urban_train_set) > len(Urban)*0.6:
        Urban_train_set_complete = True
      
    elif (Urban_train_set_complete and not Urban_valid_set_complete and w not in test_words) or (w in valid_words):
     Urban_valid_set = Urban_valid_set.append(word_subset)
     valid_words.add(w)
   
     if len(Urban_valid_set) > len(Urban)*0.2:
        Urban_valid_set_complete = True

    elif (Urban_train_set_complete and Urban_valid_set_complete) or (w in test_words):
      Urban_test_set = Urban_test_set.append(word_subset)
      test_words.add(w)

  
  # Wikipedia
  Wikipedia = unified_dataset[unified_dataset.DATASET_NAME.str.contains("Wikipedia")]
  Wikipedia_words = Wikipedia['WORD'].unique().tolist()

  Wikipedia_train_set = pd.DataFrame()
  Wikipedia_valid_set = pd.DataFrame()
  Wikipedia_test_set = pd.DataFrame()

  Wikipedia_train_set_complete=False
  Wikipedia_valid_set_complete=False

  for w in Wikipedia_words:
    word_subset = Wikipedia[Wikipedia["WORD"] == w]

    if (not Wikipedia_train_set_complete and w not in valid_words and w not in test_words) or (w in train_words):
      Wikipedia_train_set = Wikipedia_train_set.append(word_subset)
      train_words.add(w)
    
      if len(Wikipedia_train_set) > len(Wikipedia)*0.6:
        Wikipedia_train_set_complete = True
      
    elif (Wikipedia_train_set_complete and not Wikipedia_valid_set_complete and w not in test_words) or (w in valid_words):
     Wikipedia_valid_set = Wikipedia_valid_set.append(word_subset)
     valid_words.add(w)
   
     if len(Wikipedia_valid_set) > len(Wikipedia)*0.2:
        Wikipedia_valid_set_complete = True

    elif (Wikipedia_train_set_complete and Wikipedia_valid_set_complete) or (w in test_words):
      Wikipedia_test_set = Wikipedia_test_set.append(word_subset)
      test_words.add(w)
  
  
  # CHA
  CHA = unified_dataset[unified_dataset.DATASET_NAME.str.contains("CHA")]
  CHA_words = CHA['WORD'].unique().tolist()

  CHA_train_set = pd.DataFrame()
  CHA_valid_set = pd.DataFrame()
  CHA_test_set = pd.DataFrame()

  CHA_train_set_complete=False
  CHA_valid_set_complete=False

  for w in CHA_words:
    word_subset = CHA[CHA["WORD"] == w]

    if (not CHA_train_set_complete and w not in valid_words and w not in test_words) or (w in train_words):
      CHA_train_set = CHA_train_set.append(word_subset)
      train_words.add(w)
    
      if len(CHA_train_set) > len(CHA)*0.6:
        CHA_train_set_complete = True
      
    elif (CHA_train_set_complete and not CHA_valid_set_complete and w not in test_words) or (w in valid_words):
     CHA_valid_set = CHA_valid_set.append(word_subset)
     valid_words.add(w)
   
     if len(CHA_valid_set) > len(CHA)*0.2:
        CHA_valid_set_complete = True

    elif (CHA_train_set_complete and CHA_valid_set_complete) or (w in test_words):
      CHA_test_set = CHA_test_set.append(word_subset)
      test_words.add(w)
	
  
  # CODWOE
  CODWOE = unified_dataset[unified_dataset.DATASET_NAME.str.contains("CODWOE")]
  CODWOE_words = CODWOE['WORD'].unique().tolist()

  CODWOE_train_set = pd.DataFrame()
  CODWOE_valid_set = pd.DataFrame()
  CODWOE_test_set = pd.DataFrame()

  CODWOE_train_set_complete=False
  CODWOE_valid_set_complete=False

  for w in CODWOE_words:
    word_subset = CODWOE[CODWOE["WORD"] == w]

    if (not CODWOE_train_set_complete and w not in valid_words and w not in test_words) or (w in train_words):
      CODWOE_train_set = CODWOE_train_set.append(word_subset)
      train_words.add(w)
    
      if len(CODWOE_train_set) > len(CODWOE)*0.6:
        CODWOE_train_set_complete = True
      
    elif (CODWOE_train_set_complete and not CODWOE_valid_set_complete and w not in test_words) or (w in valid_words):
     CODWOE_valid_set = CODWOE_valid_set.append(word_subset)
     valid_words.add(w)
   
     if len(CODWOE_valid_set) > len(CODWOE)*0.2:
        CODWOE_valid_set_complete = True

    elif (CODWOE_train_set_complete and CODWOE_valid_set_complete) or (w in test_words):
      CODWOE_test_set = CODWOE_test_set.append(word_subset)
      test_words.add(w)
	  
  
  # Sci
  Sci = unified_dataset[unified_dataset.DATASET_NAME.str.contains("Sci-definition")]
  Sci_words = Sci['WORD'].unique().tolist()

  Sci_train_set = pd.DataFrame()
  Sci_valid_set = pd.DataFrame()
  Sci_test_set = pd.DataFrame()

  Sci_train_set_complete=False
  Sci_valid_set_complete=False

  for w in Sci_words:
    word_subset = Sci[Sci["WORD"] == w]

    if (not Sci_train_set_complete and w not in valid_words and w not in test_words) or (w in train_words):
      Sci_train_set = Sci_train_set.append(word_subset)
      train_words.add(w)
    
      if len(Sci_train_set) > len(Sci)*0.6:
        Sci_train_set_complete = True
      
    elif (Sci_train_set_complete and not Sci_valid_set_complete and w not in test_words) or (w in valid_words):
     Sci_valid_set = Sci_valid_set.append(word_subset)
     valid_words.add(w)
   
     if len(Sci_valid_set) > len(Sci)*0.2:
        Sci_valid_set_complete = True

    elif (Sci_train_set_complete and Sci_valid_set_complete) or (w in test_words):
      Sci_test_set = Sci_test_set.append(word_subset)
      test_words.add(w)
	
  
  # Webster
  Webster = unified_dataset[unified_dataset.DATASET_NAME.str.contains("Webster's Unabridged")]
  Webster_words = Webster['WORD'].unique().tolist()

  Webster_train_set = pd.DataFrame()
  Webster_valid_set = pd.DataFrame()
  Webster_test_set = pd.DataFrame()

  Webster_train_set_complete=False
  Webster_valid_set_complete=False

  for w in Webster_words:
    word_subset = Webster[Webster["WORD"] == w]

    if (not Webster_train_set_complete and w not in valid_words and w not in test_words) or (w in train_words):
      Webster_train_set = Webster_train_set.append(word_subset)
      train_words.add(w)
    
      if len(Webster_train_set) > len(Webster)*0.6:
        Webster_train_set_complete = True
      
    elif (Webster_train_set_complete and not Webster_valid_set_complete and w not in test_words) or (w in valid_words):
     Webster_valid_set = Webster_valid_set.append(word_subset)
     valid_words.add(w)
   
     if len(Webster_valid_set) > len(Webster)*0.2:
        Webster_valid_set_complete = True

    elif (Webster_train_set_complete and Webster_valid_set_complete) or (w in test_words):
      Webster_test_set = Webster_test_set.append(word_subset)
      test_words.add(w)

  
  # Hei++
  Hei = unified_dataset[unified_dataset.DATASET_NAME.str.contains("Hei\\++")]
  Hei_words = Hei['WORD'].unique().tolist()

  Hei_train_set = pd.DataFrame()
  Hei_valid_set = pd.DataFrame()
  Hei_test_set = pd.DataFrame()

  Hei_train_set_complete=False
  Hei_valid_set_complete=False

  for w in Hei_words:
    word_subset = Hei[Hei["WORD"] == w]

    if (not Hei_train_set_complete and w not in valid_words and w not in test_words) or (w in train_words):
      Hei_train_set = Hei_train_set.append(word_subset)
      train_words.add(w)
    
      if len(Hei_train_set) > len(Hei)*0.6:
        Hei_train_set_complete = True
      
    elif (Hei_train_set_complete and not Hei_valid_set_complete and w not in test_words) or (w in valid_words):
     Hei_valid_set = Hei_valid_set.append(word_subset)
     valid_words.add(w)
   
     if len(Hei_valid_set) > len(Hei)*0.2:
        Hei_valid_set_complete = True

    elif (Hei_train_set_complete and Hei_valid_set_complete) or (w in test_words):
      Hei_test_set = Hei_test_set.append(word_subset)
      test_words.add(w)

  
  # MultiRD
  MultiRD = unified_dataset[unified_dataset.DATASET_NAME.str.contains("MultiRD")]
  MultiRD_words = MultiRD['WORD'].unique().tolist()

  MultiRD_train_set = pd.DataFrame()
  MultiRD_valid_set = pd.DataFrame()
  MultiRD_test_set = pd.DataFrame()

  MultiRD_train_set_complete=False
  MultiRD_valid_set_complete=False

  for w in MultiRD_words:
    word_subset = MultiRD[MultiRD["WORD"] == w]

    if (not MultiRD_train_set_complete and w not in valid_words and w not in test_words) or (w in train_words):
      MultiRD_train_set = MultiRD_train_set.append(word_subset)
      train_words.add(w)
    
      if len(MultiRD_train_set) > len(MultiRD)*0.6:
        MultiRD_train_set_complete = True
      
    elif (MultiRD_train_set_complete and not MultiRD_valid_set_complete and w not in test_words) or (w in valid_words):
     MultiRD_valid_set = MultiRD_valid_set.append(word_subset)
     valid_words.add(w)
   
     if len(MultiRD_valid_set) > len(MultiRD)*0.2:
        MultiRD_valid_set_complete = True

    elif (MultiRD_train_set_complete and MultiRD_valid_set_complete) or (w in test_words):
      MultiRD_test_set = MultiRD_test_set.append(word_subset)
      test_words.add(w)
  
  
  # Concat 

  lexical_train = pd.concat([WordNet_train_set, Wiktionary_train_set, Urban_train_set, Wikipedia_train_set, 
                           CHA_train_set, CODWOE_train_set, Sci_train_set, Webster_train_set, Hei_train_set, MultiRD_train_set])
  lexical_train = lexical_train.drop_duplicates()

  lexical_valid = pd.concat([WordNet_valid_set, Wiktionary_valid_set, Urban_valid_set, Wikipedia_valid_set, CHA_valid_set, CODWOE_valid_set,
                          Sci_valid_set, Webster_valid_set, Hei_valid_set, MultiRD_valid_set])
  lexical_valid = lexical_valid.drop_duplicates()

  lexical_test = pd.concat([WordNet_test_set, Wiktionary_test_set, Urban_test_set, Wikipedia_test_set, CHA_test_set, CODWOE_test_set,
                          Sci_test_set, Webster_test_set, Hei_test_set, MultiRD_test_set])
  lexical_valid = lexical_valid.drop_duplicates()

  lexical_train.to_csv(os.path.join(args.output_path, "lexical_train.csv"), index = False, header=True)
  lexical_valid.to_csv(os.path.join(args.output_path, "lexical_valid.csv"), index = False, header=True)
  lexical_test.to_csv(os.path.join(args.output_path, "lexical_test.csv"), index = False, header=True)
