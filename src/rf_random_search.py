import numpy as np
import pandas as pd

from sklearn import ensemble, metrics, model_selection
import config

if __name__ == "__main__":
    df = pd.read_csv(config.MOBILE_TRAIN)
    x = df.drop("price_range", axis=1).values
    y = df.price_range.values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 31),
        "criterion": ["gini", "entropy"]
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        scoring='accuracy',
        n_jobs=-1,
        n_iter=20,
        cv=5
    )

    model.fit(x, y)

    print(f"Best Score: {model.best_score_}")

    print("Best Parameters set: ")
    print(model.best_params_)
