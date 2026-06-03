# 🏦 Machine Learning - Predicción de Cancelación de Cuentas Bancarias

## Modelo de Clasificación con CatBoost aplicado al sector financiero

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" />
  <img src="https://img.shields.io/badge/CatBoost-Clasificación-yellow" />
  <img src="https://img.shields.io/badge/Optuna-Optimización-blue" />
  <img src="https://img.shields.io/badge/Status-Completo-brightgreen" />
</p>

---

## 📋 Descripción

Proyecto de Machine Learning supervisado que desarrolla un modelo de clasificación para predecir si un cliente cancelará su cuenta bancaria (churn). Utilizando CatBoost como algoritmo principal, se construye un pipeline que abarca análisis exploratorio, manejo de desbalance de clases con SMOTENC, validación cruzada, optimización de hiperparámetros con Optuna, y análisis detallado del rendimiento mediante curvas ROC y Precisión-Recall.

---

## 🎯 Objetivo

Predecir la variable `Churn` (cancelación de cuenta) para identificar clientes con alto riesgo de abandono, permitiendo al banco implementar estrategias de retención proactivas. El desarrollo incluye:

- Exploración del dataset y análisis de correlación con la variable objetivo.
- Evaluación de técnicas de balanceo de clases (SMOTENC).
- Entrenamiento de un modelo CatBoost con manejo nativo de variables categóricas.
- Optimización de hiperparámetros con Optuna.
- Análisis del threshold óptimo mediante curva Precisión-Recall.

---

## 📂 Estructura del Repositorio

```
Machine-learning---Algoritmos-supervisados---Clasificacion-/
│
├── Predecir_cansela_cuenta_bancaria.ipynb   # Notebook con el desarrollo completo
├── churn_train.csv                          # Dataset de clientes bancarios
├── catboost_info/                           # Logs de entrenamiento de CatBoost
└── README.md                                # Documentación del proyecto
```

---

## 🔍 Metodología

### 1. Análisis Exploratorio de Datos (EDA)

- Inspección de estructura, tipos de datos y valores nulos.
- Análisis de correlación de variables numéricas con `Churn`.
- Selección de variables basada en conocimiento del negocio bancario.

**Variables utilizadas:**

| Tipo | Variables |
|---|---|
| **Numéricas** | `Contacts_Count_12_mon`, `Months_Inactive_12_mon`, `Total_Revolving_Bal`, `Total_Ct_Chng_Q4_Q1`, `Total_Trans_Ct`, `Avg_Utilization_Ratio` |
| **Categóricas** | `Education_Level`, `Marital_Status`, `Income_Category`, `Card_Category` |

### 2. Manejo de Desbalance de Clases

Se identificó un desbalance significativo en la variable objetivo (más clientes que no cancelan vs. los que sí). Se evaluó **SMOTENC** (Synthetic Minority Oversampling para datos mixtos numéricos-categóricos) como técnica de balanceo. Tras la validación cruzada, se concluyó que el modelo entrenado con los datos originales sin balancear ofrecía mejor rendimiento, por lo que se descartó el sobremuestreo.

### 3. Modelo CatBoost

Se eligió CatBoost por su capacidad nativa para manejar variables categóricas sin necesidad de codificación manual. El modelo base se configuró con `max_depth=12` y se evaluó con validación cruzada (5-Fold) usando ROC-AUC como métrica principal.

### 4. Optimización con Optuna

Se optimizaron 4 hiperparámetros mediante 10 trials:

| Hiperparámetro | Rango explorado |
|---|---|
| `iterations` | 1 - 100 |
| `depth` | 1 - 12 |
| `learning_rate` | 0.001 - 0.4 |
| `colsample_bylevel` | 0.01 - 0.1 |

La función objetivo maximiza el ROC-AUC promedio en validación cruzada.

### 5. Evaluación del Modelo

Se evaluó el modelo optimizado con múltiples métricas y visualizaciones:

| Métrica | Descripción |
|---|---|
| **Accuracy** | Proporción de predicciones correctas sobre el total |
| **Sensibilidad (Recall)** | Capacidad de detectar clientes que sí cancelan |
| **Especificidad** | Capacidad de identificar correctamente los que no cancelan |
| **ROC-AUC** | Capacidad discriminativa global entre ambas clases |

Visualizaciones generadas:
- Matriz de confusión
- Curva ROC
- Curva Precisión-Recall (para selección del threshold óptimo)
- Gráfico de importancia de variables

---

## 📊 Resultados

- **El modelo optimizado superó al modelo base** en ROC-AUC tras la optimización de hiperparámetros con Optuna.
- **SMOTENC no mejoró el rendimiento** — el modelo con datos originales generalizó mejor, posiblemente porque CatBoost maneja internamente el desbalance de forma eficiente.
- **Variables más influyentes:** `Total_Trans_Ct` (número total de transacciones) y `Total_Ct_Chng_Q4_Q1` (cambio en transacciones entre trimestres) fueron los predictores más fuertes de churn, lo cual tiene sentido: una disminución en la actividad transaccional es señal de abandono.
- **La curva Precisión-Recall** permite ajustar el threshold según la estrategia del negocio: priorizar sensibilidad (detectar más churns aunque con algunos falsos positivos) o precisión (menos alertas pero más certeras).

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.x
- **Análisis de datos:** Pandas, NumPy
- **Visualización:** Matplotlib, Seaborn
- **Modelado:** CatBoost (`CatBoostClassifier`)
- **Balanceo de clases:** imbalanced-learn (`SMOTENC`)
- **Optimización:** Optuna
- **Evaluación:** Scikit-learn (`cross_validate`, `confusion_matrix`, `roc_auc_score`, `precision_recall_curve`)

---

## 🚀 Cómo Ejecutar

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Mateo-Rua/Machine-learning---Algoritmos-supervisados---Clasificacion-.git
   cd Machine-learning---Algoritmos-supervisados---Clasificacion-
   ```

2. Instala las dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn catboost optuna imbalanced-learn
   ```

3. Abre el notebook:
   ```bash
   jupyter notebook Predecir_cansela_cuenta_bancaria.ipynb
   ```

---

## 💡 Principales Conclusiones

1. **La actividad transaccional es el mejor predictor de churn** — clientes que reducen su número de transacciones y el monto entre trimestres tienen mayor probabilidad de cancelar.

2. **CatBoost demostró ser ideal** para este dataset al manejar nativamente las variables categóricas sin necesidad de codificación, simplificando el pipeline.

3. **El sobremuestreo con SMOTENC no aportó mejora**, lo que refuerza que no siempre balancear los datos es la mejor estrategia; la evaluación empírica es clave.

4. **La curva Precisión-Recall** es más informativa que la exactitud global para seleccionar el umbral de decisión en problemas de churn con clases desbalanceadas.

5. **Optuna optimizó el modelo** encontrando una combinación de hiperparámetros que mejoró la capacidad discriminativa medida por ROC-AUC.

---


## 👤 Autor

**Mateo Rua**

[![GitHub](https://img.shields.io/badge/GitHub-Mateo--Rua-181717?logo=github)](https://github.com/Mateo-Rua)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mateo_Londoño_Rúa-0077B5?logo=linkedin)](https://www.linkedin.com/in/mateo-londono-rua117/)

---

> *Proyecto de clasificación supervisada aplicado al sector bancario para la predicción de churn, con CatBoost, optimización de hiperparámetros y análisis de curvas ROC y Precisión-Recall.*
