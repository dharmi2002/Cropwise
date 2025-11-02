import joblib
from sklearn.ensemble import RandomForestClassifier
def train(X, y, path='models/model.pkl'):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, path)
    return clf
