import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os

# 1. CONFIGURACIÓN DE PÁGINA Y DISEÑO
try:
    if os.path.exists("Logo_Clinica.png"):
        st.set_page_config(page_title="Predictor LOS | Clínica Sagrado Corazón", layout="wide", page_icon="Logo_Clinica.png")
    elif os.path.exists("logo.jpg"):
        st.set_page_config(page_title="Predictor LOS | Clínica Sagrado Corazón", layout="wide", page_icon="logo.jpg")
    else:
        st.set_page_config(page_title="Predictor LOS | Clínica Sagrado Corazón", layout="wide")
except:
    st.set_page_config(page_title="Predictor LOS | Clínica Sagrado Corazón", layout="wide")

# Aplicando paleta institucional: Azul Oscuro (#253d5b), Azul Medio (#4e6c9f), Gris (#b6b5af)
# Tipografías: Open Sans (General), Poppins (Títulos), Roboto (Datos/Textos secundarios)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&family=Poppins:wght@500;700&family=Roboto:wght@400;500&display=swap');

    /* Fondo general */
    .stApp { 
        background-color: #f4f6f9; 
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Textos y Títulos */
    h1, h2, h3 { 
        color: #253d5b; 
        font-family: 'Poppins', sans-serif; 
        font-weight: 700;
    }
    p, span, div, label { 
        color: #333333; 
        font-family: 'Roboto', sans-serif;
    }
    
    /* Botones estilo Institucional Primario */
    .stButton>button { 
        background-color: #4e6c9f; 
        color: white; 
        border-radius: 6px; 
        border: none; 
        font-weight: 600; 
        font-family: 'Open Sans', sans-serif;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: #253d5b; 
        color: white; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 10px; 
        border-bottom: 2px solid #b6b5af;
    }
    .stTabs [data-baseweb="tab"] { 
        background-color: transparent; 
        border: 2px solid transparent;
        border-radius: 5px 5px 0px 0px; 
        color: #4e6c9f; 
        padding: 10px 20px; 
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }
    .stTabs [aria-selected="true"] { 
        background-color: white; 
        color: #253d5b;
        border-top: 2px solid #253d5b;
        border-left: 2px solid #b6b5af;
        border-right: 2px solid #b6b5af;
        border-bottom: 2px solid white;
        margin-bottom: -2px;
    }
    
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
@st.cache_resource
def cargar_modelo():
    if os.path.exists('modelo_estancia_v2.pkl') and os.path.exists('columnas_modelo_v2.pkl'):
        return joblib.load('modelo_estancia_v2.pkl'), joblib.load('columnas_modelo_v2.pkl'), "V2"
    else:
        return joblib.load('modelo_estancia_v1.pkl'), joblib.load('columnas_modelo.pkl'), "V1"

try:
    modelo, columnas_entrenamiento, version_modelo = cargar_modelo()
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    st.stop()

# 3. ENCABEZADO CON LOGO
col_logo, col_tit = st.columns([1, 6])
with col_logo:
    if os.path.exists("Logo_Clinica.png"):
        st.image("Logo_Clinica.png", use_container_width=True)
    elif os.path.exists("logo.jpg"):
        st.image("logo.jpg", use_container_width=True)
    # Si no existe logo, no mostramos nada y dejamos el espacio limpio.

with col_tit:
    st.title("Sistema de Alerta Temprana y Gestión de Camas")
    st.markdown("Plataforma analítica para la predicción de estancia hospitalaria (Length of Stay).")
    if version_modelo == "V2":
        st.caption("Motor Predictivo: Versión 2.0 (Alta Precisión Clínica)")

st.markdown("---")

# 4. CREACIÓN DE PESTAÑAS
tab_individual, tab_masivo = st.tabs(["Paciente Individual", "Auditoría Masiva (Censo Diurno)"])

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
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Comorbilidades Adicionales**")
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        diabetes = st.checkbox("Diabetes")
        hipertension = st.checkbox("Hipertensión")
    with col_c2:
        cardiaca = st.checkbox("Enf. Cardíaca")
        epoc = st.checkbox("EPOC")
    with col_c3:
        hemato_onco = st.checkbox("Hemato-Oncológica")
        quimio = st.checkbox("En Quimioterapia")
    with col_c4:
        hemofilia = st.checkbox("Hemofilia")
        porfiria = st.checkbox("Porfiria")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Calcular Riesgo de Estancia", key="btn_indiv"):
        total_comorbilidades = sum([diabetes, hipertension, cardiaca, epoc, hemato_onco, quimio, hemofilia, porfiria])
        
        if version_modelo == "V2":
            datos = pd.DataFrame({
                'edad': [edad], 'Sexo': [sexo], 'PabellonIngreso': [pabellon], 'Esp': [esp], 'Dx_Agrupado': [dx],
                'Diabetes': [int(diabetes)], 'Hipertension': [int(hipertension)], 'Cardiaca': [int(cardiaca)], 
                'EPOC': [int(epoc)], 'Hemato_Onco': [int(hemato_onco)], 'Quimio': [int(quimio)], 
                'Hemofilia': [int(hemofilia)], 'Porfiria': [int(porfiria)], 
                'Total_Comorbilidades': [total_comorbilidades]
            })
        else:
            complejidad_val = 1 if total_comorbilidades > 0 else 0
            datos = pd.DataFrame({'edad': [edad], 'Sexo': [sexo], 'PabellonIngreso': [pabellon], 
                                  'Esp': [esp], 'Dx_Agrupado': [dx], 'Complejidad': [complejidad_val]})
        
        datos_finales = pd.get_dummies(datos).reindex(columns=columnas_entrenamiento, fill_value=0)
        prediccion = modelo.predict(datos_finales)[0]
        
        st.markdown("---")
        st.subheader("Resultado Clínico:")
        if "Corta" in prediccion:
            st.success(f"{prediccion} - Flujo rápido esperado. Planear alta temprana.")
        elif "Media" in prediccion:
            st.warning(f"{prediccion} - Monitoreo estándar requerido.")
        else:
            st.error(f"{prediccion} - ALERTA: Alto riesgo de ocupación prolongada. Requiere revisión de caso.")

# ==========================================
# PESTAÑA 2: CARGA MASIVA (CENSO)
# ==========================================
with tab_masivo:
    st.subheader("Proyección del Censo Actual")
    st.write("Sube el archivo Excel o CSV del censo matutino para predecir las estancias detectando pacientes desviados.")
    
    archivo_subido = st.file_uploader("Cargar Censo en formato CSV o Excel", type=["csv", "xlsx"])
    
    if archivo_subido is not None:
        try:
            # Leer archivo
            if archivo_subido.name.endswith('.csv'):
                try:
                    df_censo = pd.read_csv(archivo_subido, sep=';', encoding='latin1')
                except:
                    df_censo = pd.read_csv(archivo_subido)
            else:
                df_censo = pd.read_excel(archivo_subido)
            
            st.info(f"Se cargaron {len(df_censo)} pacientes del censo.")
            
            # Formato V2
            if version_modelo == "V2":
                if 'Dx2Nombre' in df_censo.columns and 'Dx3Nombre' in df_censo.columns:
                    df_censo['Texto_Dx'] = df_censo['Dx2Nombre'].fillna('').str.upper() + " " + df_censo['Dx3Nombre'].fillna('').str.upper()
                else:
                    df_censo['Texto_Dx'] = ""
                
                df_censo['Diabetes'] = df_censo['Texto_Dx'].str.contains('DIABETES').astype(int)
                df_censo['Hipertension'] = df_censo['Texto_Dx'].str.contains('HIPERTENSION|HTA|PRESION ALTA').astype(int)
                df_censo['Cardiaca'] = df_censo['Texto_Dx'].str.contains('CARDIAC|INFARTO|ISQUEMI|FALLA').astype(int)
                df_censo['EPOC'] = df_censo['Texto_Dx'].str.contains('EPOC|PULMONAR OBSTRUCTIVA').astype(int)
                df_censo['Hemato_Onco'] = df_censo['Texto_Dx'].str.contains('CANCER|TUMOR|LEUCEMIA|LINFOMA|NEOPLASIA|MALIGN|HEMATO').astype(int)
                df_censo['Quimio'] = df_censo['Texto_Dx'].str.contains('QUIMIO').astype(int)
                df_censo['Hemofilia'] = df_censo['Texto_Dx'].str.contains('HEMOFILIA').astype(int)
                df_censo['Porfiria'] = df_censo['Texto_Dx'].str.contains('PORFIRIA').astype(int)
                
                cols_enfermedades = ['Diabetes', 'Hipertension', 'Cardiaca', 'EPOC', 'Hemato_Onco', 'Quimio', 'Hemofilia', 'Porfiria']
                df_censo['Total_Comorbilidades'] = df_censo[cols_enfermedades].sum(axis=1)
                
                if 'Dx' in df_censo.columns:
                    df_censo['Dx_Agrupado'] = df_censo['Dx'].astype(str).str[:3]
                else:
                    df_censo['Dx_Agrupado'] = "DESCONOCIDO"
                    
                columnas_x = ['edad', 'Sexo', 'PabellonIngreso', 'Esp', 'Dx_Agrupado', 
                              'Diabetes', 'Hipertension', 'Cardiaca', 'EPOC', 'Hemato_Onco', 
                              'Quimio', 'Hemofilia', 'Porfiria', 'Total_Comorbilidades']
            else:
                if 'Dx' in df_censo.columns:
                    df_censo['Dx_Agrupado'] = df_censo['Dx'].astype(str).str[:3]
                else:
                    df_censo['Dx_Agrupado'] = "DESCONOCIDO"
                
                if 'Complejidad' not in df_censo.columns:
                    df_censo['Complejidad'] = 0
                    
                columnas_x = ['edad', 'Sexo', 'PabellonIngreso', 'Esp', 'Dx_Agrupado', 'Complejidad']
            
            # Verificar variables disponibles
            col_disponibles = [c for c in columnas_x if c in df_censo.columns]
            X_masivo = df_censo[col_disponibles].copy()
            X_masivo_encoded = pd.get_dummies(X_masivo).reindex(columns=columnas_entrenamiento, fill_value=0)
            
            df_censo['Prediccion_Estancia'] = modelo.predict(X_masivo_encoded)
            
            if 'FechaIngreso' in df_censo.columns:
                df_censo['FechaIngreso'] = pd.to_datetime(df_censo['FechaIngreso'], errors='coerce')
                df_censo['Dias_Actuales'] = (pd.Timestamp.now() - df_censo['FechaIngreso']).dt.total_seconds() / (24 * 3600)
                df_censo['Dias_Actuales'] = df_censo['Dias_Actuales'].round(1)
            else:
                df_censo['Dias_Actuales'] = 0
            
            # Alertas sin emojis
            alertas = []
            for index, row in df_censo.iterrows():
                if "Corta" in str(row['Prediccion_Estancia']) and row.get('Dias_Actuales', 0) > 3:
                    alertas.append("Desviado - Límite Superado")
                elif "Media" in str(row['Prediccion_Estancia']) and row.get('Dias_Actuales', 0) > 7:
                    alertas.append("Desviado - Límite Superado")
                else:
                    alertas.append("En progreso")
                    
            df_censo['Estado_Auditoria'] = alertas
            
            columnas_mostrar = ['edad', 'Dx_Agrupado', 'FechaIngreso', 'Dias_Actuales', 'Prediccion_Estancia', 'Estado_Auditoria']
            columnas_disponibles = [c for c in columnas_mostrar if c in df_censo.columns]
            
            st.dataframe(df_censo[columnas_disponibles], use_container_width=True)
            
            csv_exp = df_censo.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Censo con Predicciones y Alertas",
                data=csv_exp,
                file_name='Censo_Predictivo.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")}")

