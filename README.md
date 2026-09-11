# 🚀 Detección de Anomalías Transaccionales con Apache Spark MLlib

## 📋 Descripción del Proyecto
Este repositorio contiene la implementación de un modelo de clasificación binaria desarrollado en el entorno distribuido de **Apache Spark (PySpark)**. El objetivo del sistema es anticipar transacciones de ventas anómalas (riesgo operativo) utilizando aprendizaje automático.

La etiqueta de riesgo (`label`) fue construida a partir de reglas de negocio específicas: transacciones con montos superiores a $5000 realizadas en horario de madrugada (22:00 - 06:00).

## 🛠️ Stack Tecnológico
* **Big Data Framework:** Apache Spark (DataFrames API)
* **Machine Learning:** Spark MLlib (`RandomForestClassifier`, `Pipeline`, `Evaluators`)
* **Lenguajes y Librerías:** Python, Pandas (para ingesta preliminar)

## ⚙️ Arquitectura y Pipeline de Datos

1. **Ingesta y Preprocesamiento Distribuido:**
   * Carga de datos y limpieza de nulos en variables críticas.
   * Extracción de componentes temporales (hora) a partir de *timestamps*.
   
2. **Ingeniería de Características (Feature Engineering):**
   * Transformación de variables categóricas mediante `StringIndexer` (Sucursal, Producto).
   * Ensamblaje de vectores de características usando `VectorAssembler`.
   * **Prevención de Data Leakage:** Las variables `Monto_Total` y `Hora` se excluyeron estrictamente del vector de características predictivas para evitar sesgos algorítmicos, forzando al modelo a encontrar patrones estructurales subyacentes.

3. **Entrenamiento y Persistencia:**
   * Algoritmo: Random Forest (numTrees=50, maxDepth=5).
   * Orquestación de transformaciones mediante `pyspark.ml.Pipeline`.
   * Uso de `.cache()` en el DataFrame procesado para optimizar el rendimiento del clúster durante el entrenamiento.

## 📊 Resultados y Evaluación
El modelo fue validado en un subconjunto de prueba aislado (20%), obteniendo el siguiente rendimiento:

* **Exactitud Global (Accuracy):** 93.75%
* **Área bajo la curva ROC (AUC):** 0.6897

*Nota Técnica:* Dado el alto desbalance de clases inherente a la detección de anomalías, la métrica de control principal es el AUC, demostrando que el modelo tiene capacidad de discriminación matemáticamente válida por sobre una elección estocástica.

## 🚀 Optimizaciones Futuras Proyectadas
1. Implementación de **Validación Cruzada** mediante `ParamGridBuilder` para el ajuste fino de hiperparámetros.
2. Expansión del espacio de características integrando variables exógenas (ej. clima, festividades locales).
