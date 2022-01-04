# Label Encoder with XGBoost

import pandas as pd

import xgboost as xgb

from sklearn import preprocessing
from sklearn import metrics
import config


def run(fold):
    df = pd.read_csv(config.CAT_DAT_TRAIN_FOLD)
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]
    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df[col] = lbl.transform(df[col])

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(max_depth=10, n_estimators=200, n_jobs=-1)

    model.fit(x_train, df_train.target.values)

    preds = model.predict_proba(x_valid)[:, 1]

    roc_auc = metrics.roc_auc_score(df_valid.target.values, preds)

    print(f"Fold: {fold}   AUC: {roc_auc}")


if __name__ == "__main__":
    for i in range(0, 5):
        run(fold=i)
