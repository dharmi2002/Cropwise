"""Train and save a RandomForest classifier."""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_and_save(X, y, path="models/model.pkl"):
    # simple train/val split for quick feedback
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    print(f"Validation accuracy: {accuracy_score(y_val, preds):.4f}")
    print(classification_report(y_val, preds))
    joblib.dump(clf, path)
    print(f"Saved model to {path}")
    return clf
