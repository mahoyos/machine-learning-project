import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_bank_data(drop_duration=True):
    """
    Carga el dataset Bank Marketing desde UCI.
    Devuelve X (features) e y (target).
    """
    print("Descargando el dataset 'Bank Marketing' (ID 222)...")
    bank_marketing = fetch_ucirepo(id=222)
    
    X = bank_marketing.data.features.copy()
    y = bank_marketing.data.targets.copy()
    
    # Transformar el target a binario
    y = y.iloc[:, 0].replace({'no': 0, 'yes': 1})
    
    if drop_duration and 'duration' in X.columns:
        X = X.drop(columns=['duration'])
        print("Variable 'duration' omitida correctamente.")
        
    return X, y
