import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import predict_diabetes, FEATURE_EXPLANATIONS_ES, FEATURE_EXPLANATIONS_EN, TEXTS

# Pagina
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

# --- CONFIGURACIÓN DE IDIOMA ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'es'

# Botones de idioma en la esquina superior derecha
col1, col2 = st.columns([0.9, 0.1])
with col2:
    # Usamos st.empty() para evitar que los botones se dupliquen en el cambio de idioma
    en_btn = st.button("🇺🇸 EN", key="en_btn")
    es_btn = st.button("🇪🇸 ES", key="es_btn")
    
    if en_btn:
        st.session_state.lang = 'en'
        st.rerun()
    if es_btn:
        st.session_state.lang = 'es'
        st.rerun()

# Seleccionar textos según idioma
lang = st.session_state.lang
texts = TEXTS[lang]
features = FEATURE_EXPLANATIONS_ES if lang == 'es' else FEATURE_EXPLANATIONS_EN


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
st.title(texts['title'])
st.markdown(texts['subtitle'])

# --- PERFIL DEL USUARIO ---
st.header(texts['health_profile'])
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(texts['vital_signs'])
    highbp = st.radio(features['HighBP'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="highbp")
    highchol = st.radio(features['HighChol'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="highchol")
    bmi = st.slider(texts['bmi'], 15.0, 45.0, 25.0, 0.1)
    genhlth = st.selectbox(texts['gen_health'], [1,2,3,4,5])

with col2:
    st.subheader(texts['lifestyle'])
    smoker = st.radio(features['Smoker'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="smoker")
    physactivity = st.radio(features['PhysActivity'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="physactivity")
    fruits = st.radio(features['Fruits'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="fruits")
    veggies = st.radio(features['Veggies'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="veggies")
    alcohol = st.radio(features['HvyAlcoholConsump'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="alcohol")

with col3:
    st.subheader(texts['medical_history'])
    stroke = st.radio(features['Stroke'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="stroke")
    heartdisease = st.radio(features['HeartDiseaseorAttack'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="heartdisease")
    diffwalk = st.radio(features['DiffWalk'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="diffwalk")
    cholcheck = st.radio(features['CholCheck'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="cholcheck")
    nodoc = st.radio(features['NoDocbcCost'], [0, 1], format_func=lambda x: "No" if x == 0 else "Sí" if lang == 'es' else "Yes", key="nodoc")

# --- INFORMACIÓN DEMOGRÁFICA ---
st.subheader(texts['demographic'])
dcol1, dcol2, dcol3, dcol4 = st.columns(4)

with dcol1:
    age = st.selectbox(texts['age'], list(range(1,14)), format_func=lambda x: f"{x*5}-{x*5+4} {'años' if lang == 'es' else 'years'}")
with dcol2:
    education = st.selectbox(texts['education'], [1,2,3,4,5,6], format_func=lambda x: 
        ["Inicial", "Primaria", "Secundaria Incompleta", "Secundaria Completa", "Técnico", "Universitario"][x-1] if lang == 'es'
        else ["Elementary", "Middle School", "Some High School", "High School Grad", "Technical", "College"][x-1])
with dcol3:
    income = st.selectbox(texts['income'], [1,2,3,4,5,6,7,8], format_func=lambda x: 
        f"S/. {x*500} - S/. {(x+1)*500}" if x<8 else "> S/. 4000" if lang == 'es'
        else f"${x*500} - ${(x+1)*500}" if x<8 else "> $4000")
with dcol4:
    sex = st.radio(texts['gender'], [0, 1], format_func=lambda x: 
        "Femenino" if x == 0 else "Masculino" if lang == 'es' 
        else "Female" if x == 0 else "Male", key="sex")

# --- ESTADO DE SALUD RECIENTE ---
st.subheader(texts['recent_health'])
mcol1, mcol2 = st.columns(2)
with mcol1:
    menthlth = st.slider(texts['mental_health'], 0, 30, 0)
with mcol2:
    physhlth = st.slider(texts['physical_health'], 0, 30, 0)

# --- CALCULO DEL ÍNDICE DE RIESGO ---
health_risk_index = genhlth

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
    'Índice_de_Salud_General': health_risk_index
}

# --- BOTÓN DE PREDICCIÓN ---
if st.button(texts['predict_button'], type="primary"):
    with st.spinner("Analizando tu perfil de salud..." if lang == 'es' else "Analyzing your health profile..."):
        pred, prob = predict_diabetes(input_data)
        prob_percent = prob * 100

        # Determinando categoria segun los rangos
        if prob_percent <= 30:
            message = texts['no_diabetes']
            color_style = "success"
            recommendations = texts['no_diabetes_rec']
        elif 30 < prob_percent < 50:
            message = texts['prediabetes']
            color_style = "warning"
            recommendations = texts['prediabetes_rec']
        else: 
            message = texts['diabetes']
            color_style = "error"
            recommendations = texts['diabetes_rec']

    st.divider()
    st.header(texts['results_title'])

    if color_style == "success":
        st.success(message)
    elif color_style == "warning":
        st.warning(message)
    else:
        st.error(message)
    
    st.markdown(recommendations)

# --- VISUALIZACIÓN DE FACTORES CLAVE ---
st.subheader(texts['risk_factors'])

# Create a DataFrame for the risk factors
risk_data = {
    'Factor': [
        texts['gen_health'].split('(')[0].strip(), 
        features['HighBP'], 
        features['HighChol'],
        features['PhysActivity'],
        f"{features['Fruits']} & {features['Veggies']}"
    ],
    'Valor': [
        genhlth,
        1 if highbp else 0,
        1 if highchol else 0,
        1 if not physactivity else 0,
        1 if not (fruits and veggies) else 0
    ],
    'Tipo': ['Numérico' if lang == 'es' else 'Numerical', 
             'Categórico' if lang == 'es' else 'Categorical', 
             'Categórico' if lang == 'es' else 'Categorical', 
             'Categórico' if lang == 'es' else 'Categorical', 
             'Categórico' if lang == 'es' else 'Categorical']
}

risk_df = pd.DataFrame(risk_data)

# Show categorical factors separately
st.subheader(texts['categorical_risk'])
categorical_df = risk_df[risk_df['Tipo'] == ('Categórico' if lang == 'es' else 'Categorical')]
if not categorical_df.empty:
    for _, row in categorical_df.iterrows():
        status = "⚠️ ALTO RIESGO" if lang == 'es' and row['Valor'] == 1 else "✅ BAJO RIESGO" if lang == 'es' and row['Valor'] == 0 else "⚠️ HIGH RISK" if row['Valor'] == 1 else "✅ LOW RISK"
        color = "#e74c3c" if row['Valor'] == 1 else "#2ecc71"
        st.markdown(f"<span style='color:{color}; font-weight:bold'>{row['Factor']}: {status}</span>", 
                    unsafe_allow_html=True)
else:
    st.info("No se identificaron factores categóricos de riesgo" if lang == 'es' else "No categorical risk factors identified")

# Disclaimer
st.info(texts['disclaimer'])

# --- SIDEBAR ---
st.sidebar.header(texts['sidebar_title'])
st.sidebar.markdown(texts['sidebar_content'])