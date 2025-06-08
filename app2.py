# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import predict_diabetes, FEATURE_EXPLANATIONS

# --- CONFIGURACION DE PAGINA ---
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
<style>
body {
    background-color: #121212;
    color: #e0e0e0;
}

.big-font { font-size:20px !important; }
.result-box {
    border-radius:10px;
    padding:20px;
    margin-top:20px;
    border:1px solid #333;
    background-color:#1e1e1e;
}
.positive { color: #e74c3c; font-weight:bold; }
.negative { color: #2ecc71; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# --- TITULO E INTRODUCCION ---
st.title("🩺 Diabetes Risk Assessment Tool")
st.markdown("""
Esta herramienta predice el **riesgo de diabetes** basado en tus indicadores de salud.
""")

# --- PERFIL DEL USUARIO ---
st.header("Perfil de Salud del Usuario")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Signos Vitales")
    highbp = st.radio(FEATURE_EXPLANATIONS['HighBP'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="highbp")
    highchol = st.radio(FEATURE_EXPLANATIONS['HighChol'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="highchol")
    bmi = st.slider("Índice de Masa Corporal (BMI)", 15.0, 45.0, 25.0, 0.1)
    genhlth = st.selectbox("Salud General (1=Excelente, 5=Deficiente)", [1,2,3,4,5])

with col2:
    st.subheader("Estilo de Vida")
    smoker = st.radio(FEATURE_EXPLANATIONS['Smoker'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="smoker")
    physactivity = st.radio(FEATURE_EXPLANATIONS['PhysActivity'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="physactivity")
    fruits = st.radio(FEATURE_EXPLANATIONS['Fruits'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="fruits")
    veggies = st.radio(FEATURE_EXPLANATIONS['Veggies'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="veggies")
    alcohol = st.radio(FEATURE_EXPLANATIONS['HvyAlcoholConsump'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="alcohol")

with col3:
    st.subheader("Historial Médico")
    stroke = st.radio(FEATURE_EXPLANATIONS['Stroke'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="stroke")
    heartdisease = st.radio(FEATURE_EXPLANATIONS['HeartDiseaseorAttack'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="heartdisease")
    diffwalk = st.radio(FEATURE_EXPLANATIONS['DiffWalk'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="diffwalk")
    cholcheck = st.radio(FEATURE_EXPLANATIONS['CholCheck'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="cholcheck")
    nodoc = st.radio(FEATURE_EXPLANATIONS['NoDocbcCost'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key="nodoc")

# --- INFORMACIÓN DEMOGRÁFICA ---
st.subheader("Información Demográfica")
dcol1, dcol2, dcol3, dcol4 = st.columns(4)

with dcol1:
    age = st.selectbox("Rango de Edad", list(range(1,14)), format_func=lambda x: f"{x*5}-{x*5+4} años")
with dcol2:
    education = st.selectbox("Nivel Educativo", [1,2,3,4,5,6], format_func=lambda x: ["Inicial", "Primaria", "Secundaria Incompleta", "Secundaria Completa", "Técnico", "Universitario"][x-1])
with dcol3:
    income = st.selectbox("Rango de Ingreso", [1,2,3,4,5,6,7,8], format_func=lambda x: f"S/. {x*500} - S/. {(x+1)*500}" if x<8 else "> S/. 4000")
with dcol4:
    sex = st.radio("Sexo", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino", key="sex")

# --- ESTADO DE SALUD RECIENTE ---
st.subheader("Estado de Salud Reciente")
mcol1, mcol2 = st.columns(2)
with mcol1:
    menthlth = st.slider("Días de mala salud mental (últimos 30 días)", 0, 30, 0)
with mcol2:
    physhlth = st.slider("Días de mala salud física (últimos 30 días)", 0, 30, 0)

# --- CALCULO DEL ÍNDICE DE RIESGO ---
health_risk_index = genhlth * bmi

input_data = {
    'HighBP': highbp,
    'HighChol': highchol,
    'CholCheck': cholcheck,
    'Smoker': smoker,
    'Stroke': stroke,
    'HeartDiseaseorAttack': heartdisease,
    'PhysActivity': physactivity,
    'Fruits': fruits,
    'Veggies': veggies,
    'HvyAlcoholConsump': alcohol,
    'AnyHealthcare': 1,
    'NoDocbcCost': nodoc,
    'GenHlth': genhlth,
    'MentHlth': menthlth,
    'PhysHlth': physhlth,
    'DiffWalk': diffwalk,
    'Sex': sex,
    'Age': age,
    'Education': education,
    'Income': income,
    'Índice_de_Salud_General': health_risk_index  # 🔄 nombre corregido
}

# --- BOTÓN DE PREDICCIÓN ---
if st.button("🔍 Evaluar Riesgo de Diabetes", type="primary"):
    with st.spinner("Analizando tu perfil de salud..."):
        pred, prob = predict_diabetes(input_data)

    st.divider()
    st.header("🧾 Resultados del Análisis")

    if pred == 1:
        st.error(f"## ⚠️ Riesgo Alto de Diabetes: {prob*100:.1f}% de probabilidad")
        st.markdown("""
        **Recomendaciones:**
        - Consultar a un médico
        - Revisar dieta y rutina de ejercicios
        - Controlar niveles de glucosa
        """)
    else:
        st.success(f"## ✅ Riesgo Bajo de Diabetes: {prob*100:.1f}% de probabilidad")
        st.markdown("""
        **Consejos:**
        - Mantén tus hábitos saludables
        - Realiza chequeos periódicos
        - Observa cambios en tu salud
        """)

    # --- VISUALIZACIÓN DE FACTORES CLAVE ---
    st.subheader("📊 Factores de Riesgo Clave")
    fig, ax = plt.subplots(figsize=(10, 5))
    risk_factors = {
        'BMI': bmi,
        'Índice Salud': health_risk_index,
        'Alta Presión': "Sí" if highbp else "No",
        'Inactividad Física': "Sí" if not physactivity else "No",
        'Dieta Deficiente': "Sí" if not(fruits and veggies) else "No"
    }

    pd.Series(risk_factors).plot(kind='barh', ax=ax, color=["#e74c3c" if x=="Sí" or (isinstance(x,float) and x>25) else "#2ecc71" for x in risk_factors.values()])
    ax.set_title("Perfil de Riesgo Personal")
    st.pyplot(fig)

    st.info("""
    **Disclaimer:** Esta herramienta ofrece una estimación basada en datos, pero no reemplaza una evaluación médica profesional.
    """)

# --- SIDEBAR ---
st.sidebar.header("📌 Información Adicional")
st.sidebar.markdown("""
- **Modelo**: XGBoost (Recall: 90%)
- **Fuente de datos**: CDC BRFSS 2015

**Factores Considerados:**
- Índice Salud General
- Actividad Física
- Hipertensión
- Dieta y colesterol
""")