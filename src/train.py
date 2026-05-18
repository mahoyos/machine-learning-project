import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn

from src.utils import get_logger
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.loader import load_bank_data

logger = get_logger(__name__)

def build_pipeline() -> Pipeline:
    """Construye un pipeline de preprocesamiento y modelo para clasificación."""
    numeric_features = ['age', 'balance', 'day_of_week', 'campaign', 'pdays', 'previous']
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop variables not explicitly in features
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    return model

def train() -> None:
    logger.info("Cargando datos...")
    X, y = load_bank_data(drop_duration=True)
    
    logger.info("Dividiendo datos (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info("Construyendo el modelo y pipeline...")
    model_pipeline = build_pipeline()
    
    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Bank_Marketing_Logistic_Regression")
    
    with mlflow.start_run():
        logger.info("Entrenando regresión logística...")
        model_pipeline.fit(X_train, y_train)
        
        # Guardar modelo usando joblib localmente
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(model_pipeline, model_path)
        logger.info(f"Modelo guardado en {model_path}")
        
        # MLflow Logging
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.sklearn.log_model(model_pipeline, "logistic_regression_model")
        
        # Guardar dataset de testing para que puedan usarlo otras tareas
        X_test.to_csv("data/X_test.csv", index=False)
        y_test.to_csv("data/y_test.csv", index=False)
        
if __name__ == "__main__":
    train()
