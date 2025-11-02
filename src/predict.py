import joblib
def predict(x, path='models/model.pkl'):
    model = joblib.load(path)
    return model.predict(x)
