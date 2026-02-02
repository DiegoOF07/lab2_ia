import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from data_cleaning import clean_dataset, get_most_correlated, standard_scale_fit, standard_scale_transform, train_test_split
from logistic_regression import train_logistic_regression

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

    w, b, losses = train_logistic_regression(X_train, y_train, lr=0.1, epochs=500)
    
    plt.figure(figsize=(8,6))

    plt.scatter(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        alpha=0.6,
        label="Legítimo"
    )

    plt.scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        alpha=0.6,
        label="Phishing"
    )

    plt.xlabel(top2[0] + " (scaled)")
    plt.ylabel(top2[1] + " (scaled)")
    plt.title("Distribución 2D de las 2 features más correlacionadas")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(losses)
    plt.xlabel("Época")
    plt.ylabel("Log Loss")
    plt.title("Descenso del costo")
    plt.grid(True)
    plt.show()

    plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], label="Legítimo", alpha=0.6)
    plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], label="Phishing", alpha=0.6)

    x_vals = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 100)
    y_vals = -(w[0]*x_vals + b) / w[1]

    plt.plot(x_vals, y_vals, 'k--', label="Frontera logística")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.legend()
    plt.title("Regresión Logística - Frontera de decisión")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
