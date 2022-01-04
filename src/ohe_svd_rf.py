# One hot encoding, Singular value decomposition, random forest
import pandas as pd

from scipy import sparse
from sklearn import preprocessing, decomposition, ensemble, metrics

import config
import model_dispatcher
import argparse


def run(fold, model_object):
    df = pd.read_csv(config.CAT_DAT_TRAIN_FOLD)
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]
    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data)

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    svd = decomposition.TruncatedSVD(n_components=120)
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    model = model_dispatcher.models[model_object]
    model.fit(x_train, df_train.target.values)

    preds = model.predict_proba(x_valid)[:,1]

    roc_auc = metrics.roc_auc_score(df_valid.target.values, preds)

    print(f"Fold: {fold}    AUC: {roc_auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=str
    )

    args = parser.parse_args()

    for i in range(0, 5):
        run(i, args.run)

