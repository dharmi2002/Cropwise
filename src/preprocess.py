import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df
