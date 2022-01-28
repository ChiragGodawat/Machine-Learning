import numpy as np
from functools import partial
from scipy.optimize import fmin

from sklearn import metrics

class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)

        auc_score = metrics.roc_auc_score(y, predictions)

        return -1.0 * auc_score

    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)
        initial_coeff = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        self.coef_ = fmin(loss_partial, initial_coeff, disp=True)

    def predict(self, X):
        x_coeff = X * self.coef_
        predictions = np.sum(x_coeff, axis=1)
        return predictions
