"""Load and preview datasets."""
import os
import pandas as pd
from typing import Dict

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded {os.path.basename(path)} — shape: {df.shape}")
    return df

def load_all(base_path: str = "data") -> Dict[str, pd.DataFrame]:
    files = {
        "analysis": os.path.join(base_path, "crop_analysis.csv"),
        "yield": os.path.join(base_path, "crop_yield.csv"),
    }
    loaded = {}
    for k, p in files.items():
        try:
            loaded[k] = load_dataset(p)
        except Exception as e:
            print(f"❌ Could not load {k}: {e}")
    print("\nColumns summary:")
    for k, df in loaded.items():
        print(f" - {k}: {list(df.columns)}")
    return loaded

if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(f"\n--- {name} sample ---")
        print(df.head(3))
