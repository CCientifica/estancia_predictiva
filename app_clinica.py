import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# 1. CONFIGURACIÓN DE PÁGINA Y DISEÑO (ESTILO BOOTSTRAP)
st.set_page_config(page_title="Predictor LOS | Clínica", layout="wide", page_icon="🏥")

# Aplicando tu paleta de colores: Turquesa (#23d5d5), Azul Acero (#4e6c9f), Gris Cálido (#b6b5af)
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #f8f9fa; }
    
    /* Textos y Títulos */
    h1, h2, h3 { color: #4e6c9f; font-family: 'Segoe UI', Tahoma, sans-serif; font-weight: 700;}
    p { color: #333333; }
    
    /* Botones estilo Bootstrap Primario */
    .stButton>button { 
        background-color: #23d5d5; 
        color: white; 
        border-radius: 6px; 
        border: none; 
        font-weight: bold; 
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #4e6c9f; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    
    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #b6b5af; 
        border-radius: 5px 5px 0px 0px; 
        color: white; 
        padding: 10px 20px; 
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] { background-color: #4e6c9f; color: white;}
    
    /* Tarjetas de métricas */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #b6b5af;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# 2. CARGA DEL CEREBRO MATEMÁTICO
@st.cache_resource # Esto hace que cargue súper rápido
def cargar_modelo():
    return joblib.load('modelo_estancia_v1.pkl'), joblib.load('columnas_modelo.pkl')

modelo, columnas_entrenamiento = cargar_modelo()

# 3. ENCABEZADO
st.title("🏥 Sistema de Alerta Temprana y Gestión de Camas")
st.markdown("Plataforma analítica para la predicción de estancia hospitalaria (Length of Stay).")
st.markdown("---")

# 4. CREACIÓN DE PESTAÑAS
tab_individual, tab_masivo = st.tabs(["🩺 Paciente Individual", "📁 Auditoría Masiva (Censo Diurno)"])

# ==========================================
# PESTAÑA 1: PREDICCIÓN INDIVIDUAL
# ==========================================
with tab_individual:
    st.subheader("Ingreso de Paciente a Urgencias / Piso")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        edad = st.number_input("Edad del paciente", min_value=0, max_value=120, value=45)
        sexo = st.selectbox("Sexo", ["M", "F"])
    with col2:
        pabellon = st.selectbox("Servicio / Pabellón", ["URGENCIAS", "CIRUGIA", "UCI", "HOSPITALIZACION", "TRIAGE"]) 
        esp = st.text_input("Especialidad (Ej. MEDICINA INTERNA)", "MEDICINA INTERNA").upper()
    with col3:
        dx = st.text_input("Dx Principal CIE-10 (Primeras 3 letras, ej. J45)", "OTR").upper()
        # Aquí está la ilusión visual de los antecedentes
        antecedentes = st.multiselect("Antecedentes / Comorbilidades", 
                                      ["Hipertensión (HTA)", "Diabetes Mellitus", "EPOC", "Enfermedad Renal", "Falla Cardíaca", "Ninguno"])

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 Calcular Riesgo de Estancia", key="btn_indiv"):
        # Lógica temporal: Si eligió algo distinto a "Ninguno", la complejidad es 1
        complejidad_val = 1 if len(antecedentes) > 0 and "Ninguno" not in antecedentes else 0
        
        datos = pd.DataFrame({'edad': [edad], 'Sexo': [sexo], 'PabellonIngreso': [pabellon], 
                              'Esp': [esp], 'Dx_Agrupado': [dx], 'Complejidad': [complejidad_val]})
        
        datos_finales = pd.get_dummies(datos).reindex(columns=columnas_entrenamiento, fill_value=0)
        prediccion = modelo.predict(datos_finales)[0]
        
        st.markdown("---")
        st.subheader("📊 Resultado Clínico:")
        if "Corta" in prediccion:
            st.success(f"🟢 **{prediccion}** - Flujo rápido esperado. Planear alta temprana.")
        elif "Media" in prediccion:
            st.warning(f"🟡 **{prediccion}** - Monitoreo estándar requerido.")
        else:
            st.error(f"🔴 **{prediccion}** - ¡ALERTA! Alto riesgo de ocupación prolongada. Requiere revisión de caso.")

# ==========================================
# PESTAÑA 2: CARGA MASIVA (CENSO)
# ==========================================
with tab_masivo:
    st.subheader("Proyección del Censo Actual")
    st.write("Sube el archivo Excel del censo matutino para predecir las estancias y detectar pacientes desviados.")
    
    archivo_subido = st.file_uploader("Cargar Censo en formato CSV o Excel", type=["csv", "xlsx"])
    
    if archivo_subido is not None:
        try:
            # Leer archivo
            if archivo_subido.name.endswith('.csv'):
                df_censo = pd.read_csv(archivo_subido)
            else:
                df_censo = pd.read_excel(archivo_subido)
            
            st.info(f"Se cargaron {len(df_censo)} pacientes del censo.")
            
            # Asumimos que el censo tiene las mismas columnas de tu base limpia más una "FechaIngreso"
            # Si no trae FechaIngreso, no podemos calcular el desfase
            if 'FechaIngreso' in df_censo.columns:
                df_censo['FechaIngreso'] = pd.to_datetime(df_censo['FechaIngreso'], errors='coerce')
                # Calculamos cuántos días lleva el paciente hasta HOY
                df_censo['Dias_Actuales'] = (pd.Timestamp.now() - df_censo['FechaIngreso']).dt.total_seconds() / (24 * 3600)
                df_censo['Dias_Actuales'] = df_censo['Dias_Actuales'].round(1)
            else:
                st.warning("⚠️ El archivo no tiene la columna 'FechaIngreso'. No se podrá calcular el desfase de días.")
                df_censo['Dias_Actuales'] = 0
            
            # Preparar datos matemáticos
            X_masivo = df_censo[['edad', 'Sexo', 'PabellonIngreso', 'Esp', 'Dx_Agrupado', 'Complejidad']].copy()
            X_masivo_encoded = pd.get_dummies(X_masivo).reindex(columns=columnas_entrenamiento, fill_value=0)
            
            # Predecir todos a la vez
            df_censo['Prediccion_Estancia'] = modelo.predict(X_masivo_encoded)
            
            # Crear lógica de Alerta (Ej. Si lleva más de 3 días y su predicción era "Corta")
            alertas = []
            for index, row in df_censo.iterrows():
                if "Corta" in str(row['Prediccion_Estancia']) and row['Dias_Actuales'] > 3:
                    alertas.append("🚨 Desviado")
                elif "Media" in str(row['Prediccion_Estancia']) and row['Dias_Actuales'] > 7:
                    alertas.append("🚨 Desviado")
                else:
                    alertas.append("✅ En curso")
                    
            df_censo['Estado_Auditoria'] = alertas
            
            # Mostrar la tabla final hermosa
            columnas_mostrar = ['edad', 'Dx_Agrupado', 'FechaIngreso', 'Dias_Actuales', 'Prediccion_Estancia', 'Estado_Auditoria']
            columnas_disponibles = [c for c in columnas_mostrar if c in df_censo.columns]
            
            st.dataframe(df_censo[columnas_disponibles], use_container_width=True)
            
            # Botón para descargar el resultado
            csv_exp = df_censo.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Censo con Predicciones y Alertas",
                data=csv_exp,
                file_name='Censo_Predictivo.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"Ocurrió un error leyendo el archivo. Revisa que tenga las columnas correctas. Detalle: {e}")
