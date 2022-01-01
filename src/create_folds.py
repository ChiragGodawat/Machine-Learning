from sklearn import model_selection
import config
import pandas as pd


def create_k_fold_mnist_csv():

    print("Creating K Fold CSV for mnist")

    # Download original dataset from https://www.kaggle.com/oddrationale/mnist-in-csv and place it in Input folder
    df = pd.read_csv(config.MNIST_TRAIN)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.label
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (_t, _v) in enumerate(kf.split(X=df, y =y)):
        df.loc[_v, "kfold"] = f

    df.to_csv(config.MNIST_TRAIN_FOLDS, index=False)


def create_k_fold_cat_dat_csv():

    print("Creating K Fold CSV for Cat in Dat dataset")

    # Download original dataset from https://www.kaggle.com/c/cat-in-the-dat-ii/data and place it in Input folder
    df = pd.read_csv(config.CAT_DAT_TRAIN)

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_, "kfold"] = f

    df.to_csv(config.CAT_DAT_TRAIN_FOLD, index=False)
