import pandas as pd
import argparse
from sklearn import metrics
from sklearn import preprocessing

import model_dispatcher

import config


def run_cat_in_dat(fold):

    df = pd.read_csv(config.CAT_DAT_TRAIN_FOLD)

    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)

    ohe.fit(full_data)

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = model_dispatcher.models["log_res"]
    model.fit(x_train, df_train.target.values)

    preds = model.predict_proba(x_valid)[:, 1]
    auc_roc = metrics.roc_auc_score(df_valid.target.values, preds)
    print("Fold: ", fold, "\nAUC ROC Score: ", auc_roc)


def run_census(fold):
    df = pd.read_csv(config.CENSUS_TRAIN_FOLD)

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    df = df.drop(num_cols, axis=1)
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df['income'] = df.income.map(target_mapping)

    features = [f for f in df.columns if f not in ("kfold", "income")]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = model_dispatcher.models["log_res"]

    model.fit(x_train, df_train.income.values)

    preds = model.predict_proba(x_valid)[:,1]

    roc_auc = metrics.roc_auc_score(df_valid.income.values, preds)

    print(f"Fold: {fold},  ROC_AUC: {roc_auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str
    )

    args = parser.parse_args()

    if args.data == 'census':
        for i in range(0, 5):
            run_census(i)

    elif args.data == 'catdat':
        for i in range(0, 5):
            run_cat_in_dat(i)
