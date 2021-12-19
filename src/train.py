import argparse
import os

import joblib
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher
import create_folds


# Function for running model on MNIST datset
def run_mnist(fold, model):
    if not os.path.exists(config.MNIST_TRAIN_FOLDS):
        create_folds.create_k_fold_mnist_csv()

    df = pd.read_csv(config.MNIST_TRAIN_FOLDS)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)

    df_train = df_train.drop("kfold", axis=1)
    df_val = df_val.drop("kfold", axis=1)

    x_train = df_train.drop("label", axis=1).values
    y_train = df_train["label"].values

    x_val = df_val.drop("label", axis=1).values
    y_val = df_val["label"].values

    clf = model_dispatcher.models[model]

    clf.fit(x_train, y_train)

    # Calculating accuracy on validation
    preds = clf.predict(x_val)
    accuracy = metrics.accuracy_score(y_val, preds)
    print(f"Fold: {fold}, Accuracy= {accuracy}")

    # Saving Model
    joblib.dump(clf, os.path.join(config.MNIST_MODEL_OUTPUT, f"{model}_{fold}.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=str
    )
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    if args.run == "mnist":
        run_mnist(
            args.fold,
            args.model
        )
