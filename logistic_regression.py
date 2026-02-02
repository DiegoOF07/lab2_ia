import numpy as np

def sigmoid(z) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)

def log_loss(y, y_hat):
    eps = 1e-9
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

def train_logistic_regression(X, y, lr: float = 0.1, epochs: int = 500) -> tuple[np.ndarray, float, list[float]]:
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    losses = []
    
    for _ in range(epochs):
        y_hat = predict_proba(X, w, b)
        loss = log_loss(y, y_hat)
        losses.append(loss)
        error = y_hat - y
        dw = (X.T @ error) / m
        db = np.sum(error) / m
        w -= lr * dw
        b -= lr * db
    
    return w, b, losses
