import pandas as pd
import argparse
from sklearn import metrics
from sklearn import preprocessing

import model_dispatcher

import config


def run(fold, model_object):

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

    model = model_dispatcher.models[model_object]
    model.fit(x_train, df_train.target.values)

    preds = model.predict_proba(x_valid)[:, 1]
    auc_roc = metrics.roc_auc_score(df_valid.target.values, preds)
    print("Fold: ", fold, "\nAUC ROC Score: ", auc_roc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=str
    )

    args = parser.parse_args()

    for i in range(0, 5):
        run(i, args.run)
