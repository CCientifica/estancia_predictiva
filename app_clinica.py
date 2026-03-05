import streamlit as st
import pandas as pd
import joblib

# 1. Cargar el cerebro matemático y el mapa de columnas
modelo = joblib.load('modelo_estancia_v1.pkl')
columnas_entrenamiento = joblib.load('columnas_modelo.pkl')

# 2. Diseño de la página web
st.set_page_config(page_title="Predictor de Estancia", page_icon="🏥")
st.title("🏥 Predictor de Estancia Hospitalaria")
st.write("Ingrese los datos del paciente al momento del ingreso para predecir cuántos días estará ocupando una cama.")
st.markdown("---")

# 3. Formulario para ingresar datos del paciente (Simulador)
col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad del paciente", min_value=0, max_value=120, value=45)
    sexo = st.selectbox("Sexo", ["M", "F"])
    dx = st.text_input("Diagnóstico CIE-10 (Solo 3 letras, Ej. J45, I10, OTR)", "OTR").upper()

with col2:
    # Opciones comunes que vimos en tu base de datos
    pabellon = st.selectbox("Pabellón de Ingreso", ["URGENCIAS", "CIRUGIA", "TRIAGE", "UCI", "CONSULTA EXTERNA"]) 
    esp = st.text_input("Especialidad (Ej. MEDICINA GENERAL, CIRUGIA GENERAL)", "MEDICINA GENERAL").upper()
    complejidad = st.selectbox("¿Tiene comorbilidades / Diagnóstico Secundario?", ["No (0)", "Sí (1)"])

# Convertir el texto de complejidad a número
complejidad_val = 1 if "Sí" in complejidad else 0

# 4. Botón de Predicción
st.markdown("---")
if st.button("🔮 Calcular Predicción de Estancia", use_container_width=True):
    
    # Agrupar los datos ingresados
    datos_paciente = pd.DataFrame({
        'edad': [edad],
        'Sexo': [sexo],
        'PabellonIngreso': [pabellon],
        'Esp': [esp],
        'Dx_Agrupado': [dx],
        'Complejidad': [complejidad_val]
    })
    
    # Traducir los textos a la matemática que entiende el modelo
    datos_encoded = pd.get_dummies(datos_paciente)
    
    # Alinear con las columnas exactas que el modelo estudió
    datos_finales = datos_encoded.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # ¡Hacer la predicción!
    prediccion = modelo.predict(datos_finales)[0]
    
    # Mostrar el resultado con colores
    st.subheader("📊 Resultado de la Predicción:")
    if "Corta" in prediccion:
        st.success(f"🟢 **{prediccion}** - Flujo rápido esperado.")
    elif "Media" in prediccion:
        st.warning(f"🟡 **{prediccion}** - Monitoreo estándar.")
    else:
        st.error(f"🔴 **{prediccion}** - ¡Alerta! Alto riesgo de ocupación prolongada de cama.")