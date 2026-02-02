import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data_cleaning import clean_dataset, get_most_correlated, standard_scale_fit, standard_scale_transform, train_test_split
from knn import predict_knn
from logistic_regression import train_logistic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }


def load_and_prepare_data(filepath: str):
    df = pd.read_csv(filepath)
    df = clean_dataset(df)
    
    target = 'status'
    top2 = get_most_correlated(df, target)
    print(f"\nTop 2 features más correlacionadas: {top2}")
    
    X = df[top2].values
    y = df[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    mean, std = standard_scale_fit(X_train)
    X_train = standard_scale_transform(X_train, mean, std)
    X_test = standard_scale_transform(X_test, mean, std)
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, top2


def plot_data_distribution(X_train, y_train, top2):
    plt.figure(figsize=(8, 6))
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


def train_and_visualize_logistic_regression(X_train, y_train, top2):
    print("\n=== Entrenando Regresión Logística Manual ===")
    w, b, losses = train_logistic_regression(X_train, y_train, lr=0.1, epochs=500)
    
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel("Época")
    plt.ylabel("Log Loss")
    plt.title("Descenso del costo")
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label="Legítimo", alpha=0.6)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label="Phishing", alpha=0.6)
    
    x_vals = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'k--', label="Frontera logística")
    
    plt.xlabel(top2[0] + " (scaled)")
    plt.ylabel(top2[1] + " (scaled)")
    plt.legend()
    plt.title("Regresión Logística - Frontera de decisión")
    plt.grid(True)
    plt.show()
    
    return w, b, losses


def train_and_visualize_knn(X_train, y_train, X_test, y_test, top2, k: int = 3):
    print(f"\n=== Entrenando KNN Manual (k={k}) ===")
    y_pred_knn = predict_knn(X_train, y_train, X_test, k=k)
    accuracy = np.mean(y_pred_knn == y_test)
    print(f"Accuracy KNN: {accuracy:.4f}")
    
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    Z = predict_knn(X_train, y_train, grid, k=k)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label="Legítimo", alpha=0.6)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label="Phishing", alpha=0.6)
    plt.xlabel(top2[0] + " (scaled)")
    plt.ylabel(top2[1] + " (scaled)")
    plt.title(f"KNN (k={k}) Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return y_pred_knn


def compare_with_sklearn(X_train, y_train, X_test, y_test) -> None:
    print("\n=== Comparación con Sklearn ===")
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
    print("\nLogistic Regression (sklearn):")
    metrics_log = evaluate(y_test, y_pred_log)
    for metric, value in metrics_log.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nKNN (sklearn):")
    metrics_knn = evaluate(y_test, y_pred_knn)
    for metric, value in metrics_knn.items():
        print(f"  {metric}: {value:.4f}")


def display_menu() -> None:
    print("\n" + "="*50)
    print("DETECCIÓN DE PHISHING - MENÚ PRINCIPAL")
    print("="*50)
    print("1. Cargar y explorar dataset")
    print("2. Visualizar distribución de features")
    print("3. Entrenar Regresión Logística (manual)")
    print("4. Entrenar KNN (manual)")
    print("5. Comparar con implementaciones de Sklearn")
    print("6. Ejecutar pipeline completo")
    print("7. Análisis de métricas")
    print("0. Salir")
    print("="*50)


def print_metrics_analysis() -> None:
    print("\n=== ANÁLISIS DE MÉTRICAS ===")
    print("\nEn el contexto de detección de Phishing:")
    print("\n1. Falso Positivo (FP): Bloquear un sitio legítimo")
    print("   - Impacto: Usuario molesto, pérdida de confianza")
    
    print("\n2. Falso Negativo (FN): Dejar pasar un sitio de phishing")
    print("   - Impacto: Robo de credenciales, pérdida financiera, compromiso de seguridad")
    
    print("\n3. Métrica prioritaria: RECALL")
    print("   - Razón: Es más costoso dejar pasar un ataque que bloquear un sitio legítimo")
    print("   - Recall alto = Detectamos la mayoría de ataques de phishing")
    print("   - Aceptamos algunos falsos positivos para maximizar la detección")
    
    print("\n4. Balance con Precision:")
    print("   - No queremos recall 100% a costa de precision muy baja")
    print("   - Objetivo: Recall > 90% con Precision razonable (> 70%)")


def main() -> None:
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    top2 = None
    
    while True:
        display_menu()
        choice = input("\nSeleccione una opción: ")
        
        if choice == "1":
            filepath = input("Ingrese la ruta del dataset CSV (o presione Enter para './dataset_phishing.csv'): ")
            if not filepath:
                filepath = './dataset_phishing.csv'
            try:
                X_train, X_test, y_train, y_test, top2 = load_and_prepare_data(filepath)
                print("\n✓ Dataset cargado y preprocesado exitosamente")
            except Exception as e:
                print(f"\n✗ Error al cargar el dataset: {e}")
        
        elif choice == "2":
            if X_train is not None:
                plot_data_distribution(X_train, y_train, top2)
            else:
                print("\n✗ Primero debe cargar el dataset (opción 1)")
        
        elif choice == "3":
            if X_train is not None:
                train_and_visualize_logistic_regression(X_train, y_train, top2)
            else:
                print("\n✗ Primero debe cargar el dataset (opción 1)")
        
        elif choice == "4":
            if X_train is not None:
                train_and_visualize_knn(X_train, y_train, X_test, y_test, top2)
            else:
                print("\n✗ Primero debe cargar el dataset (opción 1)")
       
        elif choice == "5":
            if X_train is not None:
                compare_with_sklearn(X_train, y_train, X_test, y_test)
            else:
                print("\n✗ Primero debe cargar el dataset (opción 1)")
        
        elif choice == "6":
            filepath = input("Ingrese la ruta del dataset CSV (o presione Enter para './dataset_phishing.csv'): ")
            if not filepath:
                filepath = './dataset_phishing.csv'
            try:
                X_train, X_test, y_train, y_test, top2 = load_and_prepare_data(filepath)
                plot_data_distribution(X_train, y_train, top2)
                train_and_visualize_logistic_regression(X_train, y_train, top2)
                train_and_visualize_knn(X_train, y_train, X_test, y_test, top2)
                compare_with_sklearn(X_train, y_train, X_test, y_test)
                print("\n✓ Pipeline completo ejecutado exitosamente")
            except Exception as e:
                print(f"\n✗ Error durante la ejecución: {e}")
        
        elif choice == "7":
            print_metrics_analysis()
        
        elif choice == "0":
            print("\n¡Hasta luego!")
            break
        
        else:
            print("\n✗ Opción inválida. Por favor, seleccione una opción válida.")


if __name__ == "__main__":
    main()
