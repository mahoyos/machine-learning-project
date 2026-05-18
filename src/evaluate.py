import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

from src.utils import get_logger, save_metrics

logger = get_logger(__name__)

def evaluate() -> None:
    logger.info("Cargando modelo guardado...")
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"El modelo no existe en {model_path}. Corre train.py primero.")
        return
        
    model_pipeline = joblib.load(model_path)
    
    logger.info("Cargando datos de test...")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze() # convertir a Series
    
    logger.info("Realizando predicciones...")
    y_pred = model_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    logger.info(f"Métricas calculadas: {metrics}")
    
    # Guardar métricas localmente
    metrics_path = "reports/metrics.json"
    save_metrics(metrics, metrics_path)
    logger.info(f"Métricas guardadas en {metrics_path}")
    
    # Tratar de loaguear esto en el último run de MLFlow si es posible
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Bank_Marketing_Logistic_Regression")
    
    # Buscamos el último run reportado (debe coincidir con la corrida del workflow)
    recent_runs = mlflow.search_runs(experiment_names=["Bank_Marketing_Logistic_Regression"], order_by=["start_time DESC"], max_results=1)
    if not recent_runs.empty:
        run_id = recent_runs.iloc[0].run_id
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            logger.info(f"Métricas alojadas en MLflow Run ID: {run_id}")
    else:
        logger.warning("No se encontró ningún run activo en MLflow para registrar métricas.")

if __name__ == "__main__":
    evaluate()
