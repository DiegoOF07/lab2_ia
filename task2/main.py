import pandas as pd

from data_cleaning import clean_dataset, get_most_correlated, standard_scale_fit, standard_scale_transform, train_test_split

def main():
    df = pd.read_csv('./dataset_phishing.csv')
    df = clean_dataset(df)

    target = 'status'
    top2 = get_most_correlated(df, target)
    print("Top 2 features:", top2)

    X = df[top2].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mean, std = standard_scale_fit(X_train)
    X_train = standard_scale_transform(X_train, mean, std)
    X_test  = standard_scale_transform(X_test, mean, std)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

if __name__ == "__main__":
    main()
