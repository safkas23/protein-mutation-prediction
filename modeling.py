from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_logistic_regression(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def train_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    preds = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, preds)
