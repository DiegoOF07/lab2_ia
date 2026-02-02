import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from data_cleaning import clean_dataset, get_most_correlated, standard_scale_fit, standard_scale_transform, train_test_split
from knn import predict_knn
from logistic_regression import train_logistic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


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

    y_pred_knn = predict_knn(X_train, y_train, X_test, k=3)
    accuracy = np.mean(y_pred_knn == y_test)
    print("Accuracy KNN:", accuracy)

    x_min, x_max = X_train[:,0].min()-1, X_train[:,0].max()+1
    y_min, y_max = X_train[:,1].min()-1, X_train[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # clasificar cada punto del grid con KNN
    Z = predict_knn(X_train, y_train, grid, k=3)
    Z = Z.reshape(xx.shape)

    # mapa de color de fondo
    plt.contourf(xx, yy, Z, alpha=0.2)

    # superponer puntos reales
    plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], label="Legítimo")
    plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], label="Phishing")

    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.title("KNN (k=3) Decision Boundary")
    plt.legend()
    plt.show()

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)

    print("Logistic Regression (sklearn):")
    print(evaluate(y_test, y_pred_log))

    print("\nKNN (sklearn):")
    print(evaluate(y_test, y_pred_knn))

def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }

if __name__ == "__main__":
    main()
