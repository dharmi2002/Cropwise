"""Model training module."""
import joblib
from sklearn.ensemble import RandomForestClassifier

def train(X, y, path="models/model.pkl"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, path)
    return model
