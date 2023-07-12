
import pandas as pd
import sys
import csv
import argparse
import simpletransformers
from simpletransformers.classification import MultiLabelClassificationModel

csv.field_size_limit(sys.maxsize)
paths = {
    "lexical_test" : "../datasets/final_dataset/lexical_test.csv",
    "lexical_train" : "../datasets/final_dataset/lexical_train.csv",
    "random_test" : "../datasets/final_dataset/random_test(1).csv",
    "random_train" : "../datasets/final_dataset/random_train(1).csv"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--split', help='Split type, either lexical or random', default='lexical')
    parser.add_argument('-e','--epochs',help='Number of training epochs.', default=3, type=int)
    parser.add_argument('-o','--output_path',help='Path to save results', required=True)

    args = parser.parse_args()

    split = args.split

    df_test = pd.read_csv(paths[f"{split}_test"], engine="python")
    df_train = pd.read_csv(paths[f"{split}_train"], engine="python")

    df_test = df_test["WORD DEFINITION DATASET_NAME".split()]
    df_train = df_train["WORD DEFINITION DATASET_NAME".split()]

    counts = {}

    def convert_(s):
        s = s.replace('[', "").replace(']', "").replace("\"", "").replace("'", "")
        dsets = s.split(',')
        dsets = [i.strip() for i in dsets]
        for d in dsets:
            counts[d] = counts.get(d, 0) + 1
        return dsets

    df_test['DATASET_LIST'] = df_test.DATASET_NAME.apply(convert_)
    df_train['DATASET_LIST'] = df_train.DATASET_NAME.apply(convert_)


    dset_types = list(counts.keys())


    train_df = df_train.copy()
    train_df['input'] = train_df.apply(lambda row: f"{row['WORD']} : {row['DEFINITION']}", axis=1)
    train_df['label'] = train_df.apply(lambda row: [1 if dset_types[j] in row['DATASET_LIST'] else 0 for j in range(len(dset_types))], axis=1)

    test_df = df_test.copy()
    test_df['input'] = test_df.apply(lambda row: f"{row['WORD']} : {row['DEFINITION']}", axis=1)
    test_df['label'] = test_df.apply(lambda row: [1 if dset_types[j] in row['DATASET_LIST'] else 0 for j in range(len(dset_types))], axis=1)


    train_df = train_df.sample(frac=1, random_state=1)
    test_df = test_df.sample(frac=1, random_state=1)

    train_df = train_df[['input', 'label']]
    test_df = test_df[['input', 'label']]

    

    model = MultiLabelClassificationModel(
        "roberta",
        "roberta-base",
        num_labels=len(dset_types),
        args={
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "num_train_epochs": args.epochs,
            "save_steps" : -1,
            "use_multiprocessing": False,
            "use_multiprocessing_for_evaluation" : False,
            "output_dir" : args.output_path
        },
    )

    model.train_model(train_df)

    train_preds = model.predict(train_df['input'].tolist())
    test_preds = model.predict(test_df['input'].tolist())

    train_df['pred'] = train_preds
    test_df['pred'] = test_preds

    train_df.to_csv(f"{args.output_path}/{split}_train_predictions.csv", index=False, header=True)
    test_df.to_csv(f"{args.output_path}/{split}_test_prediction.csv", index=False, header=True)
