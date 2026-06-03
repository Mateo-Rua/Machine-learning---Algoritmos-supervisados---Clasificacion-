# 🏠 Machine Learning - Predicción de Precios de Casas

## Modelo de Regresión con Random Forest aplicado al sector inmobiliario

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-green?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Optuna-Optimización-blue" />
  <img src="https://img.shields.io/badge/Status-Completo-brightgreen" />
</p>

---

## 📋 Descripción

Proyecto de Machine Learning supervisado que desarrolla un modelo de regresión para predecir el precio de venta de viviendas en Montreal, Canadá. A partir de características físicas, estructurales y de ubicación de las propiedades, se construye un pipeline completo que abarca desde el análisis exploratorio hasta la optimización de hiperparámetros con Optuna, logrando un modelo robusto y generalizable.

---

## 🎯 Objetivo

Desarrollar un modelo predictivo que estime el precio de venta (`SalePrice`) de una vivienda con base en sus características, incluyendo:

- Exploración y análisis de correlación del dataset.
- Construcción de un pipeline de preprocesamiento + modelado con Scikit-learn.
- Evaluación del rendimiento con múltiples métricas y validación cruzada.
- Optimización de hiperparámetros con Optuna.
- Selección de las variables más relevantes con SelectKBest.

---

## 📂 Estructura del Repositorio

```
Machine-learning---Algoritmos-supervisados---Regresion-/
│
├── Predecir_precios_casas.ipynb    # Notebook con el desarrollo completo
├── house_prices.txt                # Dataset de precios de casas (Montreal, Canadá)
└── README.md                       # Documentación del proyecto
```

---

## 🔍 Metodología

### 1. Análisis Exploratorio de Datos (EDA)

- Inspección de tipos de datos y valores nulos.
- Análisis de correlación de variables numéricas con `SalePrice`.
- Feature engineering: creación de `años_construccion` a partir del año de venta y año de construcción.
- Visualización de relaciones con `pairplot`.

**Variables utilizadas:**

| Tipo | Variables |
|---|---|
| **Numéricas** | `LotFrontage`, `LotArea`, `FullBath`, `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`, `GarageArea`, `GarageCars`, `PoolArea`, `YearRemodAdd`, `años_construccion` |
| **Categóricas** | `Street`, `Utilities`, `LotConfig` |

### 2. Pipeline de Preprocesamiento

Se construyó un pipeline integrado con Scikit-learn que automatiza todo el flujo:

- **Variables categóricas** → `OneHotEncoder` (manejo de categorías desconocidas con `handle_unknown='ignore'`).
- **Variables numéricas** → `SimpleImputer` (imputación de nulos por la media).
- **Modelo** → `RandomForestRegressor` encadenado al preprocesador.

### 3. Modelo Base

Se entrenó un Random Forest inicial con `max_depth=9` y se evaluó con:

| Métrica | Descripción |
|---|---|
| **RMSE** | Error cuadrático medio (penaliza errores grandes) |
| **MAE** | Error absoluto promedio |
| **MAPE** | Error porcentual absoluto promedio |
| **R²** | Proporción de varianza explicada por el modelo |

Se realizó validación cruzada (5-Fold) para verificar la capacidad de generalización, identificando sobreajuste en el modelo base.

### 4. Optimización con Optuna

Se optimizaron 4 hiperparámetros del Random Forest mediante 10 trials:

| Hiperparámetro | Rango explorado |
|---|---|
| `n_estimators` | 1 - 5000 |
| `max_depth` | 2 - 1000 |
| `min_samples_split` | 2 - 100 |
| `min_samples_leaf` | 1 - 100 |

La función objetivo minimiza la suma de la media + desviación estándar del MAE en validación cruzada, buscando un modelo preciso y estable.

### 5. Selección de Variables (SelectKBest)

Se aplicó `SelectKBest` con `f_regression` (k=10) al pipeline optimizado para retener solo las variables con mayor poder predictivo, reduciendo ruido y mejorando la generalización.

### 6. Evaluación del Modelo Optimizado

Se compararon las métricas del modelo base vs. el modelo optimizado, evidenciando una mejora en el ajuste a los datos y menor brecha entre entrenamiento y prueba (reducción de sobreajuste). Se visualizó la importancia de cada variable seleccionada.

---

## 📊 Resultados

El modelo optimizado con Optuna + SelectKBest logró:

- **Reducción del sobreajuste** — La brecha entre scores de entrenamiento y test en validación cruzada se redujo significativamente.
- **Mejores métricas** — Mejora en MAE, MAPE y R² respecto al modelo base.
- **Variables más influyentes** — `GarageArea`, `GarageCars`, `años_construccion` y `LotArea` resultaron ser las características con mayor importancia para la predicción del precio.

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.x
- **Análisis de datos:** Pandas, NumPy
- **Visualización:** Matplotlib, Seaborn
- **Modelado:** Scikit-learn (`RandomForestRegressor`, `Pipeline`, `ColumnTransformer`)
- **Preprocesamiento:** `OneHotEncoder`, `SimpleImputer`, `SelectKBest`
- **Optimización:** Optuna
- **Evaluación:** `cross_validate`, `KFold`, `mean_absolute_error`, `r2_score`, `mean_squared_error`

---

## 🚀 Cómo Ejecutar

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Mateo-Rua/Machine-learning---Algoritmos-supervisados---Regresion-.git
   cd Machine-learning---Algoritmos-supervisados---Regresion-
   ```

2. Instala las dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn optuna
   ```

3. Abre el notebook:
   ```bash
   jupyter notebook Predecir_precios_casas.ipynb
   ```

---

## 💡 Principales Conclusiones

1. **El modelo base presentaba sobreajuste**, evidenciado por la brecha entre los scores de entrenamiento y test en la validación cruzada.

2. **Optuna encontró hiperparámetros óptimos** que equilibran complejidad y generalización, reduciendo el sobreajuste del modelo inicial.

3. **SelectKBest mejoró la eficiencia** del modelo al eliminar variables con bajo poder predictivo, manteniendo solo las 10 más relevantes.

4. **Las variables de garaje y antigüedad** (`GarageArea`, `GarageCars`, `años_construccion`) son los predictores más fuertes del precio, lo cual es coherente con el conocimiento del mercado inmobiliario.

5. **El pipeline integrado** (preprocesamiento + selección + modelo) garantiza reproducibilidad y facilita el despliegue en producción.

---

## 🔮 Mejoras Futuras

- Comparar con otros algoritmos de regresión (Gradient Boosting, XGBoost, LightGBM).
- Explorar feature engineering adicional (interacciones entre variables, transformaciones logarítmicas).
- Aumentar el número de trials de Optuna para una exploración más exhaustiva del espacio de hiperparámetros.
- Implementar serialización del modelo con `pickle` o `joblib` para despliegue.

---

## 👤 Autor

**Mateo Rua**

[![GitHub](https://img.shields.io/badge/GitHub-Mateo--Rua-181717?logo=github)](https://github.com/Mateo-Rua)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mateo_Londoño_Rúa-0077B5?logo=linkedin)](https://www.linkedin.com/in/mateo-londono-rua117/)

---

> *Proyecto de regresión supervisada aplicada al sector inmobiliario, con pipeline automatizado, optimización de hiperparámetros y selección de variables.*
