import json
import sys
import os
from src.utils import get_logger

logger = get_logger(__name__)

def validate_metrics():
    metrics_path = "reports/metrics.json"
    
    if not os.path.exists(metrics_path):
        logger.error(f"El archivo {metrics_path} no existe.")
        sys.exit(1)
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    accuracy = metrics.get('accuracy', 0)
    recall = metrics.get('recall', 0)
    
    threshold_accuracy = 0.80
    threshold_recall = 0.60  # Dependiendo del imbalance de UCI, este threshold puede que falle, pero es el requerido

    logger.info(f"Validando métricas:")
    logger.info(f"Accuracy: {accuracy:.4f} (Mínimo esperado: {threshold_accuracy})")
    logger.info(f"Recall: {recall:.4f} (Mínimo esperado: {threshold_recall})")
    
    errors = []
    if accuracy < threshold_accuracy:
        errors.append(f"Accuracy de {accuracy:.4f} no supera el threshold de {threshold_accuracy}")
    
    if recall < threshold_recall:
        errors.append(f"Recall de {recall:.4f} no supera el threshold de {threshold_recall}")
        
    if errors:
        logger.error("La validación falló debido a los siguientes motivos:")
        for e in errors:
            logger.error(" - " + e)
        sys.exit(1)
    else:
        logger.info("El modelo superó exitosamente la validación de métricas.")
        sys.exit(0)

if __name__ == "__main__":
    validate_metrics()
