import numpy as np
from pandas import DataFrame

def clean_dataset(df: DataFrame) -> DataFrame:
    df = df.drop(columns=['url'])
    df['status'] = df['status'].replace({'legitimate': 0, 'phishing': 1}).astype(int)
    return df

def get_most_correlated(df: DataFrame, target: str) -> list[str]:
    y = df[target]
    X = df.drop(columns=[target])

    X = X.select_dtypes(include='number')
    X = X.loc[:, X.std() != 0]

    corr = X.corrwith(y).dropna()
    top2 = corr.abs().sort_values(ascending=False).head(2).index.tolist()
    return top2

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)

    cut = int(n * (1 - test_size))
    train_idx = idx[:cut]
    test_idx  = idx[cut:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standard_scale_fit(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    return mean, std

def standard_scale_transform(X, mean, std):
    return (X - mean) / std

