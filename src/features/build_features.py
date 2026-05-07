import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para Feature Engineering del Bank Marketing Dataset.
    Se integra perfectamente en pipelines de Scikit-Learn.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Evitar modificar el DataFrame original
        X_out = X.copy()
        
        # 1. PDAYS: 999 significa que no fue contactado antes.
        # Creamos una variable binaria y arreglamos el valor atípico
        if 'pdays' in X_out.columns:
            X_out['contacted_before'] = (X_out['pdays'] != 999).astype(int)
            # Reemplazar 999 por -1 para que el modelo lineal no asuma 
            # que 999 días es "mucho tiempo de algo bueno"
            X_out.loc[X_out['pdays'] == 999, 'pdays'] = -1
            
        # 2. PREVIOUS: Contactos previos en otras campañas
        if 'previous' in X_out.columns:
            X_out['has_previous_contact'] = (X_out['previous'] > 0).astype(int)
            
        # 3. AGE: Agrupación demográfica
        if 'age' in X_out.columns:
            X_out['is_senior'] = (X_out['age'] >= 60).astype(int)
            X_out['is_young'] = (X_out['age'] <= 30).astype(int)
            
        # 4. BALANCE: Distinguir si la cuenta está en sobregiro (negativo)
        if 'balance' in X_out.columns:
            X_out['has_negative_balance'] = (X_out['balance'] < 0).astype(int)
            X_out['has_high_balance'] = (X_out['balance'] > 5000).astype(int)
            
        # 5. POUTCOME: Juntar categorías poco informativas o limpiar unknowns si es necesario
        # En este caso, 'unknown' y 'other' suelen comportarse igual de mal.
        if 'poutcome' in X_out.columns:
            X_out['poutcome_clean'] = X_out['poutcome'].replace('other', 'unknown')
            X_out = X_out.drop(columns=['poutcome'])
            
        return X_out
