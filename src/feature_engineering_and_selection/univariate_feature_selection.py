from sklearn.feature_selection import chi2, f_classif, f_regression, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.datasets import fetch_california_housing


class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        if problem_type == 'classification':
            valid_scoring = {
                "f1_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression
            }

        if scoring not in valid_scoring:
            raise Exception("Invalid Scoring function")

        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features*100)
            )
        else:
            raise Exception("Invalid Type of Feature")

    def fit(self, X, y):
        return self.selection.fit(X, y)

    def transform(self, X):
        return self.selection.transform(X)

    def fit_transform(self, X, y):
        return self.selection.fit_transform(X,y)


if __name__ == "__main__":
    data = fetch_california_housing()
    X = data['data']
    y = data["target"]
    ufs = UnivariateFeatureSelection(n_features=1, problem_type="regression", scoring="f_regression")
    ufs.fit(X, y)
    X_transform = ufs.transform(X)
    print(X_transform)