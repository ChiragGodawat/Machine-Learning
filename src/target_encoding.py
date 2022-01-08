import copy
import pandas as pd

from sklearn import metrics, preprocessing
import xgboost as xgb

import config


def mean_target_encoding(data):
    df = copy.deepcopy(data)
    num_cols = [
        'fnlwgt',
        'age',
        'capital.gain',
        'capital.loss',
        'hours.per.week'
    ]

    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }

    df['income'] = df.income.map(target_mapping)

    features = [f for f in df.columns if f not in ("kfold", "income") and f not in num_cols]

    for col in features:
        if col not in num_cols:
            df[col] = df[col].astype(str).fillna("NONE")
            lbl_enc = preprocessing.LabelEncoder()

            lbl_enc.fit(df[col])

            df[col] = lbl_enc.transform(df[col])

    encoded_dfs = []

    for fold in range(0,5):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        for column in features:
            map_dict = dict(df_train.groupby(column)['income'].mean())
            df_valid[column + "_enc"] = df_valid[column].map(map_dict)

        encoded_dfs.append(df_valid)
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df


def run(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [f for f in df.columns if f not in ('kfold', 'income')]

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(max_depth=7)

    model.fit(x_train, df_train.income.values)

    preds = model.predict_proba(x_valid)[:, 1]

    roc_auc = metrics.roc_auc_score(df_valid.income.values, preds)

    print(f"Fold: {fold}   ROC_AUC: {roc_auc}")


if __name__ == "__main__":
    df = pd.read_csv(config.CENSUS_TRAIN_FOLD)

    df = mean_target_encoding(df)

    for i in range(0, 5):
        run(df, i)
