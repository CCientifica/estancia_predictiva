import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import io
import plotly.express as px
from streamlit_option_menu import option_menu

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
    if os.path.exists('modelo_estancia_v4.pkl') and os.path.exists('columnas_modelo_v4.pkl'):
        return joblib.load('modelo_estancia_v4.pkl'), joblib.load('columnas_modelo_v4.pkl'), "V4"
    elif os.path.exists('modelo_estancia_v3.pkl') and os.path.exists('columnas_modelo_v3.pkl'):
        return joblib.load('modelo_estancia_v3.pkl'), joblib.load('columnas_modelo_v3.pkl'), "V3"
    elif os.path.exists('modelo_estancia_v2.pkl') and os.path.exists('columnas_modelo_v2.pkl'):
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
    if version_modelo == "V4":
        st.caption("Motor Predictivo: Versión 4.0 (Regresor Numérico Exacto en Días)")
    elif version_modelo == "V3":
        st.caption("Motor Predictivo: Versión 3.0 (Alta Resolución con Minería Clínica)")
    elif version_modelo == "V2":
        st.caption("Motor Predictivo: Versión 2.0 (Alta Precisión Clínica)")

st.markdown("---")

# 4. CREACIÓN DE NAVEGACIÓN (ESTILO BOOTSTRAP / NATIVA)
selected = option_menu(
    menu_title=None,
    options=["Paciente Individual", "Auditoría Masiva"],
    icons=["person-bounding-box", "bar-chart-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa", "border-top": "2px solid #b6b5af", "border-bottom": "2px solid #b6b5af"},
        "icon": {"color": "#4e6c9f", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#e0e0e0", "font-family": "Poppins", "font-weight": "600", "color": "#4e6c9f"},
        "nav-link-selected": {"background-color": "#253d5b", "color": "white"},
    }
)

# ==========================================
# PESTAÑA 1: PREDICCIÓN INDIVIDUAL
# ==========================================
if selected == "Paciente Individual":
    st.subheader("Ingreso de Paciente a Urgencias / Piso")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        edad = st.number_input("Edad del paciente", min_value=0, max_value=120, value=45)
        sexo = st.selectbox("Sexo", ["M", "F"])
    with col2:
        pabellon = st.selectbox("Servicio / Pabellón", ["URGENCIAS", "CIRUGIA", "UCI", "HOSPITALIZACION", "TRIAGE"]) 
        esp = st.text_input("Especialidad (Ej. MEDICINA INTERNA)", "MEDICINA INTERNA").upper()
    with col3:
        dx = st.text_input("Dx Principal CIE-10 (Hasta 4 letras, ej. J450)", "OTR").upper()[:4]
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Comorbilidades Adicionales**")
    
    col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
    with col_c1:
        diabetes = st.checkbox("Diabetes")
        hipertension = st.checkbox("Hipertensión")
    with col_c2:
        cardiaca = st.checkbox("Enf. Cardíaca")
        epoc = st.checkbox("EPOC")
    with col_c3:
        hemato_onco = st.checkbox("Cáncer / Oncológica")
        quimio = st.checkbox("En Quimioterapia")
    with col_c4:
        hemofilia = st.checkbox("Hemofilia")
        porfiria = st.checkbox("Porfiria")
    with col_c5:
        renal = st.checkbox("Enfermedad Renal")
        vih = st.checkbox("VIH / SIDA")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Calcular Riesgo de Estancia", key="btn_indiv"):
        total_comorbilidades = sum([diabetes, hipertension, cardiaca, epoc, hemato_onco, quimio, hemofilia, porfiria, renal, vih])
        
        if version_modelo in ["V2", "V3", "V4"]:
            datos = pd.DataFrame({
                'edad': [edad], 'Sexo': [sexo], 'PabellonIngreso': [pabellon], 'Esp': [esp], 'Dx_Agrupado': [dx],
                'Diabetes': [int(diabetes)], 'Hipertension': [int(hipertension)], 'Cardiaca': [int(cardiaca)], 
                'EPOC': [int(epoc)], 'Hemato_Onco': [int(hemato_onco)], 'Quimio': [int(quimio)], 
                'Hemofilia': [int(hemofilia)], 'Porfiria': [int(porfiria)], 
                'Renal': [int(renal)], 'VIH': [int(vih)], 
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
        
        if version_modelo == "V4":
            dias_estimados = float(prediccion)
            min_dias = max(1, int(dias_estimados))
            max_dias = int(dias_estimados) + 2 if dias_estimados % 1 > 0 else int(dias_estimados) + 1
            
            if dias_estimados > 7:
                st.error(f"🚨 **ALERTA - Riesgo de Ocupación Prolongada.**\n\n📍 Estancia estimada: **{dias_estimados:.1f} días**.")
            elif dias_estimados > 3:
                st.warning(f"🟡 **PRECAUCIÓN - Monitoreo estándar requerido.**\n\n📍 Estancia estimada: **{dias_estimados:.1f} días**.")
            else:
                st.success(f"🟢 **ÓPTIMO - Flujo rápido esperado.**\n\n📍 Estancia estimada: **{dias_estimados:.1f} días**.")
        else:
            if "Corta" in prediccion:
                st.success(f"{prediccion} - Flujo rápido esperado. Planear alta temprana.")
            elif "Media" in prediccion:
                st.warning(f"{prediccion} - Monitoreo estándar requerido.")
            else:
                st.error(f"{prediccion} - ALERTA: Alto riesgo de ocupación prolongada. Requiere revisión de caso.")

# ==========================================
# PESTAÑA 2: CARGA MASIVA (CENSO)
# ==========================================
if selected == "Auditoría Masiva":
    st.subheader("Proyección del Censo Actual")
    st.write("Sube el archivo Excel o CSV del censo matutino para predecir las estancias detectando pacientes desviados.")
    
    archivo_subido = st.file_uploader("Cargar Censo en formato CSV o Excel", type=["csv", "xlsx", "xls"])
    
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
            
            # --- ESTANDARIZACIÓN DE MÚLTIPLES ESTRUCTURAS DE CENSO ---
            # Limpiar espacios ocultos y hacer un mapeo más robusto (ignora mayúsculas/minúsculas)
            columnas_nuevas = []
            for col in df_censo.columns:
                c_clean = str(col).strip().upper()
                if c_clean == 'EDAD': columnas_nuevas.append('edad')
                elif c_clean in ['FECINGRESO', 'FECHA INGRESO', 'FECHA INGRESO']: columnas_nuevas.append('FechaIngreso')
                elif c_clean in ['PABACTUAL', 'PABELLON ACTUAL', 'PABELLONACTUAL']: columnas_nuevas.append('PabellonIngreso_Primario')
                elif c_clean in ['PABINGRESO', 'PABELLON DE INGRESO', 'PABELLONINGRESO']: columnas_nuevas.append('PabellonIngreso_Secundario')
                elif c_clean in ['ESPECTRATANTE', 'ESPECIALIDAD TRATANTE']: columnas_nuevas.append('Esp')
                elif c_clean in ['DX1', 'COD. DIAG.', 'COD. DIAG']: columnas_nuevas.append('Dx')
                elif c_clean == 'CAMA': columnas_nuevas.append('CAMA')
                elif c_clean == 'SEXO': columnas_nuevas.append('Sexo')
                elif c_clean in ['FECHANACIMIENTO', 'FECHA DE NACIMIENTO']: columnas_nuevas.append('FechaNacimiento')
                elif c_clean in ['IDENTIFICACION', 'NUM DOC', 'TFCEDU', 'NOINGRESO']: columnas_nuevas.append('Identificacion')
                elif c_clean in ['PACIENTE', 'NOMBRE1', 'NOMBRES']: columnas_nuevas.append('Paciente_Nombre')
                elif c_clean in ['APELLIDO1', 'APELLIDOS']: columnas_nuevas.append('Paciente_Apellido')
                else: columnas_nuevas.append(str(col).strip()) # Mantiene el original limpio
                
            df_censo.columns = columnas_nuevas
            
            # Consolidar el Pabellón de prioridad (PabActual gana)
            if 'PabellonIngreso_Primario' in df_censo.columns:
                df_censo['PabellonIngreso'] = df_censo['PabellonIngreso_Primario']
            elif 'PabellonIngreso_Secundario' in df_censo.columns:
                df_censo['PabellonIngreso'] = df_censo['PabellonIngreso_Secundario']
                
            # Mapeo Inteligente de Camas (Pabellones / Servicios)
            def clasificar_pabellon(cama):
                if pd.isna(cama): return "DESCONOCIDO"
                cama = str(cama).strip().upper()
                
                # Reglas Hemo-Oncología (Piso 2)
                if cama in [str(i) for i in range(201, 223)]: 
                    return "SEGUNDO PISO (HEMATO-ONCOLOGIA)"
                
                # Reglas Piso 3
                tercer_piso = [str(i) for i in range(306, 323)] + ["301PA", "301PB", "302P", "303P", "304P", "305P", "323P", "324P", "325P", "326P"]
                if cama in tercer_piso: return "TERCER PISO"
                
                # Reglas Piso 4
                cuarto_piso = [str(i) for i in range(401, 416)] + ["416A", "416B", "417A", "417B", "418A", "418B", "419A", "419B", "420A", "420B", "421", "422", "423A", "423B", "424A", "424B", "425A", "425B", "426"]
                if cama in cuarto_piso: return "CUARTO PISO"
                
                # Reglas Cuidado Crítico
                if cama.startswith("UCI") or cama.startswith("UCE"): 
                    return "CUIDADO CRITICO"
                    
                # Reglas Urgencias
                if cama.startswith(("AU", "PAS", "PED", "REA", "SI", "YES", "H2")): 
                    return "URGENCIAS"
                    
                return "DESCONOCIDO"
            
            # Asignar Pabellón a partir de la Cama si el formato no trajo PabellonIngreso explícito
            if 'PabellonIngreso' not in df_censo.columns and 'CAMA' in df_censo.columns:
                df_censo['PabellonIngreso'] = df_censo['CAMA'].apply(clasificar_pabellon)
            elif 'PabellonIngreso' in df_censo.columns and 'CAMA' in df_censo.columns:
                # Rellena espacios nulos del pabellon usando la cama
                mascara_vacios = df_censo['PabellonIngreso'].isna() | (df_censo['PabellonIngreso'] == '')
                df_censo.loc[mascara_vacios, 'PabellonIngreso'] = df_censo.loc[mascara_vacios, 'CAMA'].apply(clasificar_pabellon)
            
            if 'edad' in df_censo.columns:
                # Extraer solo los números de campos como "57 A"
                df_censo['edad'] = df_censo['edad'].astype(str).str.replace(r'\D', '', regex=True)
                df_censo['edad'] = pd.to_numeric(df_censo['edad'], errors='coerce').fillna(50)
            elif 'FechaNacimiento' in df_censo.columns:
                df_censo['FechaNacimiento'] = pd.to_datetime(df_censo['FechaNacimiento'], errors='coerce')
                df_censo['edad'] = (pd.Timestamp.now() - df_censo['FechaNacimiento']).dt.total_seconds() / (3600 * 24 * 365.25)
                df_censo['edad'] = df_censo['edad'].fillna(50)
            else:
                df_censo['edad'] = 50
                st.toast("⚠️ El censo no contenía la edad de los pacientes. El sistema asumió 50 años como promedio preventivo.", icon="⚠️")
                    
            if 'Sexo' not in df_censo.columns:
                df_censo['Sexo'] = 'M'
            # -------------------------------------------------------------
            
            # Formato V2 / V3 / V4
            if version_modelo in ["V2", "V3", "V4"]:
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
                df_censo['Renal'] = df_censo['Texto_Dx'].str.contains('RENAL|NEFRO').astype(int)
                df_censo['VIH'] = df_censo['Texto_Dx'].str.contains('VIH|SIDA|INMUNODEFICIENCIA HUMANA').astype(int)
                
                cols_enfermedades = ['Diabetes', 'Hipertension', 'Cardiaca', 'EPOC', 'Hemato_Onco', 'Quimio', 'Hemofilia', 'Porfiria', 'Renal', 'VIH']
                df_censo['Total_Comorbilidades'] = df_censo[cols_enfermedades].sum(axis=1)
                
                if 'Dx' in df_censo.columns:
                    # Permitir 4 dígitos para V3/V4, 3 para V2 (mantener compatibilidad)
                    digits = 4 if version_modelo in ["V3", "V4"] else 3
                    df_censo['Dx_Agrupado'] = df_censo['Dx'].astype(str).str[:digits]
                else:
                    df_censo['Dx_Agrupado'] = "DESCONOCIDO"
                    
                columnas_x = ['edad', 'Sexo', 'PabellonIngreso', 'Esp', 'Dx_Agrupado', 
                              'Diabetes', 'Hipertension', 'Cardiaca', 'EPOC', 'Hemato_Onco', 
                              'Quimio', 'Hemofilia', 'Porfiria', 'Renal', 'VIH', 'Total_Comorbilidades']
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
            
            # Alertas dinámicas
            alertas = []
            riesgo_prolongado = 0
            
            for index, row in df_censo.iterrows():
                if version_modelo == "V4":
                    dias_pred = float(row['Prediccion_Estancia'])
                    dias_actuales = row.get('Dias_Actuales', 0)
                    
                    if dias_pred > 7:
                        riesgo_prolongado += 1
                        
                    # La lógica real de desviación: si ya lleva más días de los que la IA predijo
                    if dias_actuales > dias_pred:
                        alertas.append("Desviado - Límite Superado")
                    else:
                        alertas.append("Normal - Dentro del límite estimado")
                    
                    # Formatear la salida para la tabla como número exacto
                    df_censo.at[index, 'Prediccion_Estancia'] = f"{dias_pred:.1f} días"

                else:
                    if "Prolongada" in str(row['Prediccion_Estancia']):
                        riesgo_prolongado += 1
                        
                    if "Corta" in str(row['Prediccion_Estancia']) and row.get('Dias_Actuales', 0) > 3:
                        alertas.append("Desviado - Límite Superado")
                    elif "Media" in str(row['Prediccion_Estancia']) and row.get('Dias_Actuales', 0) > 7:
                        alertas.append("Desviado - Límite Superado")
                    else:
                        alertas.append("Normal - Dentro del límite estimado")
                    
            df_censo['Estado_Auditoria'] = alertas
            
            # --- SECCIÓN DE GRAFICACIÓN PROFESIONAL ---
            st.markdown("### 📊 Tablero de Control de Riesgo y Ocupación")
            
            # Métricas Superiores
            total_pacientes = len(df_censo)
            desviados = sum(1 for a in alertas if "Desviado" in a)
            
            # Usando columnas para los KPI
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(label="Pacientes Ingresados", value=total_pacientes)
            kpi2.metric(label="Desviados (Superaron Límite Estimado)", value=desviados, delta=f"{(desviados/total_pacientes)*100:.1f}%", delta_color="inverse")
            kpi3.metric(label="Predicción Estancias Prolongadas", value=riesgo_prolongado)
            
            # Gráficos con Plotly
            st.markdown("<br>", unsafe_allow_html=True)
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Gráfico de Pastel: Distribución de Riesgos Predictivos
                riesgo_counts = df_censo['Prediccion_Estancia'].value_counts().reset_index()
                riesgo_counts.columns = ['Categoría de Estancia', 'Cantidad']
                
                fig_pie = px.pie(riesgo_counts, values='Cantidad', names='Categoría de Estancia', 
                                 title='Distribución de Riesgo de Estancia',
                                 color='Categoría de Estancia',
                                 color_discrete_map={
                                     'Estancia Corta (<3 dias)': '#4e6c9f', 
                                     'Estancia Media (3 a 7 dias)': '#b6b5af', 
                                     'Estancia Prolongada (>7 dias)': '#253d5b'
                                 })
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                # Gráfico de Barras: Pacientes con Alerta por Pabellón
                if 'PabellonIngreso' in df_censo.columns:
                    desviados_df = df_censo[df_censo['Estado_Auditoria'].str.contains("Desviado")]
                    pabellon_counts = desviados_df['PabellonIngreso'].value_counts().reset_index()
                    pabellon_counts.columns = ['Pabellón', 'Pacientes Desviados']
                    
                    fig_bar = px.bar(pabellon_counts, x='Pabellón', y='Pacientes Desviados',
                                     title='Alarmas de Desviación por Pabellón',
                                     color_discrete_sequence=['#253d5b'])
                    
                    fig_bar.update_layout(xaxis_tickangle=-45, margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("El gráfico de Pabellones requiere la columna 'PabellonIngreso' o 'PabActual'.")

            # Mini Gráficos por Categoría / Servicio Individual
            st.markdown("---")
            st.markdown("### 🏥 Radiografía por Servicio Individual (Pabellón Actual)")
            
            if 'PabellonIngreso' in df_censo.columns:
                pabellones = [p for p in df_censo['PabellonIngreso'].unique() if pd.notna(p) and str(p).strip() != ""]
                cols_pab = st.columns(3)
                
                for i, pab in enumerate(pabellones):
                    df_pab = df_censo[df_censo['PabellonIngreso'] == pab]
                    if len(df_pab) == 0: continue
                    
                    estado_counts = df_pab['Estado_Auditoria'].value_counts().reset_index()
                    estado_counts.columns = ['Estado', 'Cant']
                    
                    fig_mini = px.pie(estado_counts, values='Cant', names='Estado', hole=0.5,
                                      title=f'{str(pab)[:20]} ({len(df_pab)})',
                                      color='Estado',
                                      color_discrete_map={
                                          'Normal - Dentro del límite estimado': '#b6b5af', 
                                          'Desviado - Límite Superado': '#253d5b'
                                      })
                    fig_mini.update_traces(textinfo='value')
                    fig_mini.update_layout(showlegend=False, margin=dict(t=30, b=10, l=10, r=10), height=200, title_x=0.5, title_font_size=13)
                    
                    with cols_pab[i % 3]:
                        st.plotly_chart(fig_mini, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📋 Detalle Nominal de Auditoría")
            columnas_mostrar = ['Identificacion', 'Paciente_Nombre', 'Paciente_Apellido', 'CAMA', 'edad', 'PabellonIngreso', 'Dx_Agrupado', 'FechaIngreso', 'Dias_Actuales', 'Prediccion_Estancia', 'Estado_Auditoria']
            columnas_disponibles = [c for c in columnas_mostrar if c in df_censo.columns]
            
            df_mostrar = df_censo[columnas_disponibles]
            
            # Resaltar en rojo los pacientes desviados usando pandas styling
            def aplicar_estilos(styler):
                styler.apply(lambda x: ['background-color: #ffdce0' if 'Desviado' in str(x.get('Estado_Auditoria', '')) else '' for _ in x], axis=1)
                
                # Colores institucionales para encabezados
                styler.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#253d5b'), ('color', 'white'), ('font-weight', 'bold')]}
                ])
                return styler
                
            styled_df = df_mostrar.style.pipe(aplicar_estilos)
            
            st.dataframe(styled_df, use_container_width=True)
            
            # --- EXPORTAR A EXCEL (ESTILIZADO) ---
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                styled_df.to_excel(writer, index=False, sheet_name='Censo_Auditable')
                
                # Auto-ajustar ancho de columnas en Excel
                worksheet = writer.sheets['Censo_Auditable']
                for idx, col in enumerate(df_mostrar.columns):
                    max_len = max(df_mostrar[col].astype(str).map(len).max(), len(str(col))) + 2
                    worksheet.column_dimensions[chr(65 + idx)].width = max_len
                    
            excel_data = buffer.getvalue()
            
            st.download_button(
                label="📥 Descargar Censo Completo (Excel con Colores)",
                data=excel_data,
                file_name='Censo_Predictivo_Institucional.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                type='primary'
            )
            
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")