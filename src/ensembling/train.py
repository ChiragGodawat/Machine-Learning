# Compare Ensembling by avg technique and optimizing auc techniques (Optimizing weights to output of a model)
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics, model_selection
from optimize_auc import OptimizeAUC

X, y = make_classification(n_samples=10000, n_features=25)

xfold1, xfold2, yfold1, yfold2 = model_selection.train_test_split(X, y, test_size=0.5, stratify=y)

logreg = linear_model.LogisticRegression()
rf = RandomForestClassifier()
xgbc = xgb.XGBClassifier()

logreg.fit(xfold1, yfold1)
rf.fit(xfold1, yfold1)
xgbc.fit(xfold1, yfold1)

pred_logreg = logreg.predict_proba(xfold2)[:, 1]
pred_rf = rf.predict_proba(xfold2)[:, 1]
pred_xgbc = xgbc.predict_proba(xfold2)[:, 1]

avg_pred = (pred_rf + pred_xgbc + pred_logreg)/3

fold2_pred = np.column_stack((
    pred_logreg,
    pred_rf,
    pred_xgbc,
    avg_pred
))

auc_fold2 = []

for i in range(fold2_pred.shape[1]):
    auc = metrics.roc_auc_score(yfold2, fold2_pred[:, i])
    auc_fold2.append(auc)

print(f"Fold-2 -  LR AUC: {auc_fold2[0]}")
print(f"Fold-2 -  RF AUC: {auc_fold2[1]}")
print(f"Fold-2 -  XGB AUC: {auc_fold2[2]}")
print(f"Fold-2 -  AVG AUC: {auc_fold2[3]}")


# now fit on fold 2 and predict on fold 1


logreg = linear_model.LogisticRegression()
rf = RandomForestClassifier()
xgbc = xgb.XGBClassifier()

logreg.fit(xfold2, yfold2)
rf.fit(xfold2, yfold2)
xgbc.fit(xfold2, yfold2)

pred_logreg = logreg.predict_proba(xfold1)[:, 1]
pred_rf = rf.predict_proba(xfold1)[:, 1]
pred_xgbc = xgbc.predict_proba(xfold1)[:, 1]

avg_pred = (pred_rf + pred_xgbc + pred_logreg)/3

fold1_pred = np.column_stack((
    pred_logreg,
    pred_rf,
    pred_xgbc,
    avg_pred
))

auc_fold1 = []

for i in range(fold1_pred.shape[1]):
    auc = metrics.roc_auc_score(yfold1, fold1_pred[:, i])
    auc_fold1.append(auc)

print(f"Fold-2 -  LR AUC: {auc_fold1[0]}")
print(f"Fold-2 -  RF AUC: {auc_fold1[1]}")
print(f"Fold-2 -  XGB AUC: {auc_fold1[2]}")
print(f"Fold-2 -  AVG AUC: {auc_fold1[3]}")
print()

opt = OptimizeAUC()

# Removing the average column
opt.fit(fold1_pred[:, :-1], yfold1)
opt_preds_fold2 = opt.predict(fold2_pred[:, :-1])
auc = metrics.roc_auc_score(yfold2, opt_preds_fold2)
print(f"Optimized AUC, Fold 2={auc}")
print(f"Coefficients = {opt.coef_}")


opt = OptimizeAUC()

# Removing the average column
opt.fit(fold2_pred[:, :-1], yfold2)
opt_preds_fold1 = opt.predict(fold1_pred[:, :-1])
auc = metrics.roc_auc_score(yfold1, opt_preds_fold1)
print(f"Optimized AUC, Fold 1={auc}")
print(f"Coefficients = {opt.coef_}")
