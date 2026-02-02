# Laboratorio 2 - Inteligencia Artificial

## Descripción

Este repositorio contiene el Laboratorio 2 del curso de Inteligencia Artificial. El proyecto implementa algoritmos de clasificación para detección de sitios de phishing utilizando regresión logística y k-vecinos más cercanos (KNN). Se incluyen implementaciones manuales de ambos algoritmos y se comparan con las implementaciones de scikit-learn.

## Características

- Implementación manual de regresión logística con gradient descent
- Implementación manual de algoritmo KNN
- Visualización de datos y resultados de clasificación
- Comparación con implementaciones de scikit-learn
- Interfaz interactiva para explorar diferentes opciones
- Análisis de features más correlacionadas con el target

## Dependencias

Para ejecutar este proyecto, necesitas instalar las siguientes dependencias:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Cómo correrlo

1. Asegúrate de tener el dataset `dataset_phishing.csv` en el directorio raíz
2. Ejecuta el script principal:

```bash
python main.py
```

3. Sigue el menú interactivo para explorar las diferentes opciones:

   - **Opción 1**: Cargar y explorar el dataset
   - **Opción 2**: Visualizar distribución de features más correlacionadas
   - **Opción 3**: Entrenar regresión logística (implementación manual)
   - **Opción 4**: Entrenar KNN (implementación manual)
   - **Opción 5**: Comparar resultados con scikit-learn
   - **Opción 6**: Ejecutar pipeline completo
   - **Opción 7**: Salir

## Estructura del proyecto

- `main.py`: Script principal con interfaz interactiva y visualizaciones
- `knn.py`: Implementación manual del algoritmo KNN
- `logistic_regression.py`: Implementación manual de regresión logística
- `data_cleaning.py`: Funciones para limpieza y preprocesamiento de datos
- `dataset_phishing.csv`: Dataset de sitios web para detección de phishing

## Dataset

El proyecto utiliza un dataset de sitios web clasificados como "legitimate" o "phishing" basado en diferentes características del URL y contenido del sitio.
