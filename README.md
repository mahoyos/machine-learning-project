# Bank Marketing Prediction - End-to-End MLOps Pipeline

Este proyecto implementa un pipeline completo de Machine Learning Operations (MLOps) para predecir la suscripción a un depósito a plazo (Term Deposit) mediante modelos de Scikit-Learn basados en datos del repositorio UCI *(Bank Marketing)*.

## 🏗 Arquitectura del Proyecto

```text
project/
│
├── .github/workflows/   # Definición de GitHub Actions CI/CD
├── data/                # Scripts de carga y datos intermedios
├── notebooks/           # Notebooks Jupyter para EDA y análisis de modelos
├── src/                 # Scripts Python modulares para el Pipeline
│   ├── utils.py         # Utilidades comunes (logging, guardado)
│   ├── train.py         # Lógica de preprocesamiento, build pipeline y entrenamiento
│   ├── evaluate.py      # Lógica de evaluación y métricas sobre set de prueba
│   └── validate.py      # Script de Quality Gate (falla si accuracy/recall es muy bajo)
├── models/              # Artefactos (modelo local persistido model.pkl)
├── reports/             # Métricas en formato JSON
├── mlruns/              # Entorno local de tracking de MLflow
├── app.py               # Aplicación inferencia en Streamlit
├── requirements.txt     # Dependencias de Python
└── README.md            # Documentación
```

## ⚙️ Características MLOps Implementadas

1. **Separación de roles:** Notebooks para exploración, scripts modulares en `src/` para producción.
2. **Experiment Tracking:** MLflow está configurado para llevar un registro automático e histórico de todas las métricas, parámetros y modelos.
3. **CI/CD Pipeline (GitHub Actions):** Construye el entorno, instala dependencias, corre el pipeline de train y evaluate, ejecuta un chequeo estático de calidad de métricas (failsafe condition), y sube resultados como artifacts.
4. **Validación Automática:** Se agregó un umbral (threshold) de validación para `accuracy` y `recall`. Si el nuevo modelo cae por debajo de estos números, el build de CI falla para prevenir despliegues de regresión en calidad.
5. **Inferencia Interactiva:** Streamlit sirve como UI para testear las predicciones y probabilidad sobre el modelo con interacciones visuales simples y rápidas.

## 🚀 Cómo ejecutar localmente

1. **Clonar este repositorio** y entrar en la carpeta.
2. **Crear entorno virtual** (opcional pero recomendado) e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. **Entrenar y evaluar** el modelo en tu entorno local de la siguiente forma (en orden):
   ```bash
   python src/train.py
   python src/evaluate.py
   python src/validate.py
   ```
   *(El primer script exportará el modelo .pkl y el dataset de test. El segundo sacará predicciones, guardará metrics y lo enviará también a MLflow)*.

## 📊 Explorar Resultados en MLflow

Para ver los experimentos, configuraciones pasadas, y el historial de iteraciones, levanta el servidor local de interfaz gráfica mediante:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Luego navega a `http://127.0.0.1:5000` en tu explorador predeterminado.

## 🌐 Iniciar la Demo de Streamlit

Asegúrate de que haya un modelo serializado previamente entrenado en `models/model.pkl`. Luego corre:
```bash
streamlit run app.py
```
Abre `http://localhost:8501` y juega con los tensores (features del banco de datos) para ver qué predice la regresión logística.