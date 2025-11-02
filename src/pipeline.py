"""End-to-end pipeline: load -> clean -> preprocess -> train -> evaluate."""
from src.data_loader import load_all
from src.preprocess import clean, select_common_features, encode_and_scale
from src.train_model import train_and_save
from src.evaluate_model import evaluate
import pandas as pd

def main():
    data = load_all("data")
    if "analysis" not in data or "yield" not in data:
        print("Required datasets not found. Ensure crop_analysis.csv and crop_yield.csv are in /data.")
        return
    train_df = clean(data["analysis"])
    test_df = clean(data["yield"])
    # align
    try:
        X_train, y_train, X_test, y_test, features = select_common_features(train_df, test_df)
    except Exception as e:
        print("Error aligning features:", e)
        return
    print("Selected features:", features)
    X_train_enc, X_test_enc = encode_and_scale(X_train, X_test)
    # Train
    clf = train_and_save(X_train_enc, y_train, path="models/model.pkl")
    # Evaluate (if test has target)
    evaluate("models/model.pkl", X_test_enc, y_test)

if __name__ == "__main__":
    main()
