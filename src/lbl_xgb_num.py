# Label Encoder with XGBoost with Numerical data

import pandas as pd

import xgboost as xgb

from sklearn import preprocessing
from sklearn import metrics
import config


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

    model = xgb.XGBClassifier()

    model.fit(x_train, df_train.income.values)

    preds = model.predict_proba(x_valid)[:, 1]

    roc_auc = metrics.roc_auc_score(df_valid.income.values, preds)

    print(f"Fold: {fold}  ROC_AUC:{roc_auc}")


if __name__ == "__main__":
    for i in range(0, 5):
        run_census_adult(fold=i)
