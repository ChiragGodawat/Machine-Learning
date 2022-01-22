import numpy as np
import pandas as pd

from sklearn import ensemble, metrics, model_selection
from src import config

if __name__ == "__main__":
    df = pd.read_csv(config.MOBILE_TRAIN)
    x = df.drop("price_range", axis=1).values
    y = df.price_range.values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200, 250, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 11, 15],
        "criterion": ["gini", "entropy"]
    }

    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=-1,
        cv=5
    )

    model.fit(x, y)

    print(f"Best Score: {model.best_score_}")

    print("Best Parameters set: ")
    print(model.best_params_)
