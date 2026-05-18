import streamlit as st
import pandas as pd
import joblib
import os

# Configuración inicial de Streamlit
st.set_page_config(page_title="Bank Marketing Prediction App", layout="centered")

st.title("💰 Predicción de Marketing Bancario")
st.write("Esta aplicación predice si un cliente suscribirá un depósito a plazo (term deposit).")

MODEL_PATH = "models/model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error("No se encontró el modelo entrenado. Por favor, corre el entrenamiento (`python src/train.py`) primero.")
else:
    st.sidebar.header("Ingresar datos del cliente")
    
    # Inputs básicos (puedes agregar los restantes según tu esquema final)
    # ['age', 'balance', 'day_of_week', 'campaign', 'pdays', 'previous']
    # ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    age = st.sidebar.slider("Edad", 18, 100, 30)
    balance = st.sidebar.number_input("Saldo (Balance)", value=500)
    day_of_week = st.sidebar.slider("Día de la semana (1=Lun...)", 1, 5, 1) # Simplificado numérico
    campaign = st.sidebar.number_input("Contactos en campaña actual", value=1, min_value=1)
    pdays = st.sidebar.number_input("Días desde el último contacto previo (-1 si nunca)", value=-1)
    previous = st.sidebar.number_input("Número de contactos previos a esta campaña", value=0, min_value=0)
    
    job = st.sidebar.selectbox("Trabajo", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.sidebar.selectbox("Estado civil", ['divorced', 'married', 'single', 'unknown'])
    education = st.sidebar.selectbox("Educación", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'])
    default = st.sidebar.selectbox("Crédito en suspensión", ['no', 'yes', 'unknown'])
    housing = st.sidebar.selectbox("Préstamo hipotecario", ['no', 'yes', 'unknown'])
    loan = st.sidebar.selectbox("Préstamo personal", ['no', 'yes', 'unknown'])
    contact = st.sidebar.selectbox("Medio de contacto", ['cellular', 'telephone'])
    month = st.sidebar.selectbox("Mes de último contacto", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    poutcome = st.sidebar.selectbox("Resultado de la campaña anterior", ['failure', 'nonexistent', 'success'])
    
    # Predecir
    if st.button("Predecir Suscripción"):
        # Construir dataframe con única fila
        input_data = pd.DataFrame([{
            'age': age,
            'balance': balance,
            'day_of_week': day_of_week,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'poutcome': poutcome
        }])
        
        # Predicción
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        st.subheader("Resultado de la Predicción")
        if pred == 1:
            st.success(f"El modelo predice: **Suscribirá** (Clase: 1). Probabilidad {prob:.2%}.")
        else:
            st.warning(f"El modelo predice: **No suscribirá** (Clase: 0). Probabilidad {prob:.2%}.")

