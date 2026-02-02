import numpy as np

def predict_knn(X_train, y_train, X_test, k=3):
    preds = []

    for x in X_test:
        dists = np.sqrt(np.sum((X_train - x)**2, axis=1))

        idx = np.argsort(dists)[:k]

        labels = y_train[idx]
        pred = np.round(np.mean(labels))  # 0 o 1
        preds.append(pred)

    return np.array(preds)
