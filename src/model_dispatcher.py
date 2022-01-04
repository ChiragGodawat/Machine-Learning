from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model


models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "rf": ensemble.RandomForestClassifier(),
    "log_res": linear_model.LogisticRegression()
}