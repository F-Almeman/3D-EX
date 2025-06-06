# 3D-EX

This repository is created for the work **3D-EX: A Unified Dataset of Definitions and Dictionary Examples** that is submitted to RANLP 2023.

**3D-EX** datasset is available here  to download [3D-EX.csv](https://drive.google.com/uc?export=download&id=1ZjuRUn6KZPaXMVYecZ5IYDIRiB5VuEsR) and [3D-EX.json](https://drive.google.com/uc?export=download&id=1gnRFRKISVxLVGhwpOWg6ZfjYNdW6Nad-).

# 1. Data preparation

## 1.1 Datasets

In this section we introduce set of dictionaries/enclopedias that define English words and may provide some examples to understand words in context. We have 10 datasets whic are: CHA (Oxford Dictionary), CODWOE, Hei++, MultiRD, Urban, Sci-definition, Webster's Unabridged, Wikipedia, Wiktionary, WordNet. These datasets are organized in 2 folders [incomplete_datasets](https://github.com/F-Almeman/3D-EX/tree/main/datasets/incomplete_datasets) if they provide examples and [complete_datasets](https://github.com/F-Almeman/3D-EX/tree/main/datasets/complete_datasets) if they do not. 

| DATASET  | DEFINITION | EXAMPLE |
| ------------- | ------------- | ------------- |
| [CHA (Oxford Dictionary)](https://miulab.myds.me:5001/sharing/lWPBRc8hG) | <ul><li>- [x] </li> | <ul><li>- [x] </li> | 
| [CODWOE](https://codwoe.atilf.fr/)  | <ul><li>- [x] </li>  | <ul><li>- [x] </li> | 
| [Hei++](https://sapienzanlp.github.io/generationary-web/) | <ul><li>- [x] </li> | <ul><li>- [ ] </li> | 
| [MultiRD](https://github.com/thunlp/MultiRD) | <ul><li>- [x] </li> | <ul><li>- [ ] </li> | 
| [Urban](https://github.com/machelreid/vcdm) | <ul><li>- [x] </li> | <ul><li>- [x] </li> | 
| [Sci-definition](https://huggingface.co/datasets/talaugust/sci-definition) | <ul><li>- [x] </li> | <ul><li>- [x] </li> | 
| [Webster's Unabridged](https://github.com/Vocaby/dictionaryminer) | <ul><li>- [x] </li> | <ul><li>- [ ] </li> |
| [Wikipedia](https://github.com/machelreid/vcdm) | <ul><li>- [x] </li> | <ul><li>- [x] </li> | 
| [Wiktionary <br /> (Preprocessed by FEWS)](https://nlp.cs.washington.edu/fews/) | <ul><li>- [x] </li>  | <ul><li>- [x] </li> |
| WordNet <br /> (We extracted all WN lemmas that have examples from the Natural Language Tool Kit (NLTK) in Python and we ignored words that are less than 4 characters.) | <ul><li>- [x] </li> | <ul><li>- [x] </li> | 

These datasets are processed as following:
1. Converting txt and json files to csv.
2. Removing special tokens and any noisy characters such as tab sign and replacement character �.
3. Removing rows where their definitions have more than 10% non alphanumeric characters.
4. Removing rows that have empty words or definitions.
5. Removing duplicate rows within each dataset or split.
6. Lower-casing WORD, DEFINITION, and EXAMPLE columns.
7. Adding a new column "DATASET_NAME".
8. Adding a new column "SPLIT_TYPE" if the dataset belongs to a specific split (train/test/validation split)
9. MultiRD : removing noisy or uninformative definitions such as "see synonyms at" and "often used in the plural".
10. Sci-definition: each term has 10 scientific abstracts, we extracted the sentences that include this target term from these abstracts to be (examples) for the word. We excluded keywords sentences and any sentence has more than 10% non alphanumeric characters.
11. Wiktionary: some definitions include the time where words were coined such as (first attested in the late 16th century) and (from 16 c). These parts in the definitions were deleted. 
12. Urban: after all these cleaning steps, we found that there were still some noisy definitions in Urban dictionary. So we built a binary classifier “RoBERTa” where positive examples are from (Wikipedia, CHA and Wordnet) and negative examples are those noisy Urban definitions. All the classification details are available here in this [Google Colab Notebook](https://colab.research.google.com/drive/1SVBEgm3jFleCLM_sJXQK1VIU65sAB3VU?usp=sharing).


## 1.2 Unify the datasets
	
[unify_datasets_v1.py](https://github.com/F-Almeman/3D-EX/blob/main/src/unify_datasets_v1.py) aims to unify number datasets in one dataset. It takes as input number of datasets (a folder containing these datasets as csv files), and returns a big csv file with all the datasets concatenated. It is used here to unifiy the complete datasets in one file and also the incomplete datasets that do not have examples in one file. The output files are available to be downloded [complete_unified_dataset.csv](https://drive.google.com/uc?export=download&id=1sQRZyn8nt4a3ooQdJ7hS8ZE4OVNi8MPb),and [incomplete_unified_dataset.csv](https://drive.google.com/uc?export=download&id=1vBZU82bWECB17Wm_UTXDf7_z-3lphzHZ).
```
python3 src/unify_datasets_v1.py -i datasets/complete_datasets/ -o datasets/complete_unified_dataset.csv

python3 src/unify_datasets_v1.py -i datasets/incomplete_datasets/ -o datasets/incomplete_unified_dataset.csv
```
After that, [unify_datasets_v2.py](https://github.com/F-Almeman/3D-EX/blob/main/src/unify_datasets_v2.py) is used to unify complete_unified_dataset.csv and incomplete_unified_dataset.csv and generate one unified dataset that has the following columns: WORD, DEFINITION, EXAMPLES_LIST, DATASETS_LIST. The generated unified_dataset **3D-EX** is available here [3D-EX.csv](https://drive.google.com/uc?export=download&id=1ZjuRUn6KZPaXMVYecZ5IYDIRiB5VuEsR) and [3D-EX.json](https://drive.google.com/uc?export=download&id=1gnRFRKISVxLVGhwpOWg6ZfjYNdW6Nad-).
	
```
python3 src/unify_datasets_v2.py -c datasets/complete_unified_dataset.csv -i datasets/incomplete_unified_dataset.csv -o datasets/3D-EX.csv
```

## 1.3 Dataset splitting
	
Two splits are generated for **3D-EX**: random split and lexical split, where all instances of a given word do not appear across splits. [generate_splits.py](https://github.com/F-Almeman/3D-EX/blob/main/src/generate_splits.py) creates 3 random splits (60% train, 20% validation, and 20% test), and 3 lexical splits (60% train, 20% validation, and 20% test). 

```
python3 src/generate_splits.py datasets/unified_dataset.csv -o datasets/splits
```

[random_train.csv](https://drive.google.com/uc?export=download&id=10dgLkNhrf8KgVfElUTKoJoexH__bceZ4) <br />
[random_valid.csv](https://drive.google.com/uc?export=download&id=1lgb8Ecn_LBkwPuE5pAze0I8LoY4FtuOq) <br />
[random_test.csv](https://drive.google.com/uc?export=download&id=1EK4R_NOX1bqo1g4x9vfliP0bPjh2q34s) <br />

[lexical_train.csv](https://drive.google.com/uc?export=download&id=1tATK85HmqPW-SxKqHXyNRwNxoulrs-vn) <br />
[lexical_valid.csv](https://drive.google.com/uc?export=download&id=1x86gOZBfJLc95DiaRkyieGfGUtCG7yvp) <br />
[lexical_test.csv](https://drive.google.com/uc?export=download&id=1toafKZjqb-vVuLAKpZNyyFSIeIhlPMtz) <br />

# 4. Intrinisic evaluation
	
## 4.1 Source classification
This task consists of a multi-lable classification, he goal is to, given a <word,definition> instance, predict its original sources. [source_classification.py](https://github.com/F-Almeman/3D-EX/blob/main/src/source_classification.py) trains a RoBERTa-base on either our "lexical" or "random" splits, and then computes final predictions for both test and train splits. <br /><br />
-s: split type, either "lexical" or "random" (default=lexical)<br />
-e: number of training epochs (default=3)<br />
-o: output directory to save checkpoints and final predictions<br />
```
python3 src/source_classification.py -s lexical -e 3 -o "path_to_output_directory"
```
	
## 4.2 Reverse dictinary
	
This task is a ranking problem in which, given a definition, the task is to retrieve a ranked list of the most relevant words. [reverse_dictionary.py](https://github.com/F-Almeman/3D-EX/blob/main/src/reverse_dictionary.py) computes the Mean Reciprocal Rank (MRR), which rewards the position of the first correct result in a ranked list of outcomes, using SBERT models (all-MiniLM-L6-v2 , all-distilroberta-v1 and all-mpnet-base-v2) or instructor (using three variants of Instructor for encoding words and definitions). <br />
-m: model type, it is either all-MiniLM-L6-v2 , all-distilroberta-v1, all-mpnet-base-v2, or Instructor (defult="all-MiniLM-L6-v2").<br /> 
-d: data input file (WORD, DEFINITION).<br /> 
-wi: word instruction (defult="no").<br /> 
-di: definition instruction (defult="no").<br /> 
```
python3 src/reverse_dictionary.py -m model -d random_test.csv -wi "no" -di "Represent this dictionary definition" 
```
# Licensing Information

This is not a single dataset, therefore each subset has its own license (the collection itself does not have additional restrictions).


# Citation information
```
@inproceedings{almeman-etal-2023-3d,
    title = "3{D}-{EX}: A Unified Dataset of Definitions and Dictionary Examples",
    author = "Almeman, Fatemah  and
      Sheikhi, Hadi  and
      Espinosa Anke, Luis",
    editor = "Mitkov, Ruslan  and
      Angelova, Galia",
    booktitle = "Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.ranlp-1.8/",
    pages = "69--79",
}
```
