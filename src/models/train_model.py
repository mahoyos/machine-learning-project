from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from src.features.build_features import BankFeatureEngineer
import numpy as np

def build_logistic_pipeline(X, class_weight='balanced', max_iter=1000):
    """
    Construye un Pipeline de Scikit-Learn que incluye:
    1. Feature Engineering
    2. Preprocesamiento (Escalado + OneHotEncoding)
    3. Modelo LogisticRegression
    """
    # Instanciamos el transformador para saber qué columnas resultan
    fe = BankFeatureEngineer()
    X_fe = fe.fit_transform(X)
    
    numeric_features = X_fe.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    categorical_features = X_fe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Pipelines internos para números y categorías
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", one_hot_encoder)
    ])
    
    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
    # Pipeline Completo
    pipeline = Pipeline(steps=[
        ('engineer', BankFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, class_weight=class_weight, max_iter=max_iter))
    ])
    
    return pipeline

def optimize_threshold(pipeline, X_val, y_val):
    """
    Dado un modelo entrenado, encuentra el umbral de decisión (0-1) 
    que maximiza el F1-Score para equilibrar Precisión y Recall.
    """
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Calcular F1-scores ignorando divisiones por cero
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    return best_threshold, f1_scores[best_idx]
