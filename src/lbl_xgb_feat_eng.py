# Label Encoder with XGBoost with Numerical data and feature engineering

import pandas as pd

import xgboost as xgb

import itertools
from sklearn import preprocessing
from sklearn import metrics
import config


def feature_engineering(df, cat_cols):
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[
            :,
            c1 + "_" + c2
        ] = df[c1].astype(str) + "_" + df[c2].astype(str)

    return df


def run_census_adult(fold):
    df = pd.read_csv(config.CENSUS_TRAIN_FOLD)

    num_cols = [
        'fnlwgt',
        'age',
        'capital.gain',
        'capital.loss',
        'hours.per.week'
    ]

    target_mapping = {
        ">50K": 1,
        "<=50K": 0
    }

    df['income'] = df.income.map(target_mapping)

    cat_cols = [c for c in df.columns if c not in num_cols and c not in ("kfold", "income")]

    df = feature_engineering(df, cat_cols)
    features = [f for f in df.columns if f not in ("kfold", "income")]

    for col in features:
        if col not in num_cols:
            df[col] = df[col].astype(str).fillna("NONE")
            lbl_enc = preprocessing.LabelEncoder()

            lbl_enc.fit(df[col])

            df[col] = lbl_enc.transform(df[col])

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(max_depth=7)

    model.fit(x_train, df_train.income.values)

    preds = model.predict_proba(x_valid)[:, 1]

    roc_auc = metrics.roc_auc_score(df_valid.income.values, preds)

    print(f"Fold: {fold}  ROC_AUC:{roc_auc}")


if __name__ == "__main__":
    for i in range(0, 5):
        run_census_adult(fold=i)
