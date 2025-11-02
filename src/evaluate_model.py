"""Evaluate a saved model on test data."""
import joblib
from sklearn.metrics import accuracy_score, classification_report

def evaluate(model_path: str, X_test, y_test):
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    if y_test is None:
        print("No ground-truth target in test dataset; showing top predictions:")
        unique, counts = np.unique(preds, return_counts=True)
        print(dict(zip(unique, counts)))
        return
    print("Test accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
