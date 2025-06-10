import pickle
import pandas as pd

def load_model():
    """Load trained model from file"""
    with open('models/modelo_optimo_1.0_9_100_7_0.01_0.7_42.sav', 'rb') as f:
        model = pickle.load(f)
        # Verificar las características del modelo
        print("Características del modelo:", model.get_booster().feature_names)
        return model

def predict_diabetes(input_data):
    # Convertir input_data en DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Lista actualizada con los nombres EXACTOS que espera el modelo
    expected_features = [
        'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
        'Income', 'Índice_de_Salud_General'  # Nombre CORRECTO en español
    ]
    
    # Verificar y reordenar
    missing = [f for f in expected_features if f not in input_df.columns]
    if missing:
        raise ValueError(f"Faltan características: {missing}")
    
    input_df = input_df[expected_features]
    
    # Cargar modelo y predecir
    model = load_model()
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    return pred, proba

# Diccionarios de idiomas
FEATURE_EXPLANATIONS_ES = {
    'HighBP': "¿Tiene presión arterial alta?",
    'HighChol': "¿Tiene el colesterol alto?",
    'CholCheck': "¿Ha realizado un chequeo de colesterol en los últimos 5 años?",
    'Smoker': "¿Es fumador?",
    'Stroke': "¿Ha tenido un derrame cerebral?",
    'HeartDiseaseorAttack': "¿Ha tenido enfermedad cardíaca o ataque al corazón?",
    'PhysActivity': "¿Realiza actividad física?",
    'Fruits': "¿Consume frutas regularmente?",
    'Veggies': "¿Consume verduras regularmente?",
    'HvyAlcoholConsump': "¿Consume alcohol en exceso?",
    'AnyHealthcare': "¿Tiene acceso a atención médica?",
    'NoDocbcCost': "¿Ha evitado ir al médico por razones de costo?",
    'GenHlth': "¿Cómo calificaría su salud general?",
    'MentHlth': "Días de mala salud mental (últimos 30 días)",
    'PhysHlth': "Días de mala salud física (últimos 30 días)",
    'DiffWalk': "¿Tiene dificultad para caminar?",
    'Sex': "Género",
    'Age': "Edad",
    'Education': "Nivel educativo",
    'Income': "Ingresos"
}

FEATURE_EXPLANATIONS_EN = {
    'HighBP': "Do you have high blood pressure?",
    'HighChol': "Do you have high cholesterol?",
    'CholCheck': "Have you had a cholesterol check in the past 5 years?",
    'Smoker': "Are you a smoker?",
    'Stroke': "Have you ever had a stroke?",
    'HeartDiseaseorAttack': "Have you had heart disease or heart attack?",
    'PhysActivity': "Do you engage in physical activity?",
    'Fruits': "Do you regularly consume fruits?",
    'Veggies': "Do you regularly consume vegetables?",
    'HvyAlcoholConsump': "Do you engage in heavy alcohol consumption?",
    'AnyHealthcare': "Do you have healthcare coverage?",
    'NoDocbcCost': "Have you avoided seeing a doctor due to cost?",
    'GenHlth': "How would you rate your general health?",
    'MentHlth': "Days of poor mental health (last 30 days)",
    'PhysHlth': "Days of poor physical health (last 30 days)",
    'DiffWalk': "Do you have difficulty walking?",
    'Sex': "Gender",
    'Age': "Age",
    'Education': "Education level",
    'Income': "Income"
}

# Diccionarios para textos generales
TEXTS = {
    'es': {
        'title': "🩺 Herramienta de Evaluación de Riesgo de Diabetes",
        'subtitle': "Esta herramienta predice el **riesgo de diabetes** basado en tus indicadores de salud.",
        'health_profile': "Perfil de Salud del Usuario",
        'vital_signs': "Signos Vitales",
        'lifestyle': "Estilo de Vida",
        'medical_history': "Historial Médico",
        'bmi': "Índice de Masa Corporal (BMI)",
        'gen_health': "Salud General (1=Excelente, 5=Deficiente)",
        'demographic': "Información Demográfica",
        'age': "Rango de Edad",
        'education': "Nivel Educativo",
        'income': "Rango de Ingreso",
        'gender': "Género",
        'recent_health': "Estado de Salud Reciente",
        'mental_health': "Días de mala salud mental (últimos 30 días)",
        'physical_health': "Días de mala salud física (últimos 30 días)",
        'predict_button': "🔍 Evaluar Riesgo de Diabetes",
        'results_title': "🧾 Resultados del Análisis",
        'no_diabetes': "## ✅ No tiene diabetes",
        'prediabetes': "## ⚠️ Tiene prediabetes",
        'diabetes': "## 🚨 Tiene diabetes",
        'no_diabetes_rec': "**Consejos:**\n- Mantén tus hábitos saludables\n- Realiza chequeos cada 5 años",
        'prediabetes_rec': "**Recomendaciones:**\n- Consulte a un médico para confirmar\n- Monitoree sus niveles de glucosa\n- Ajuste dieta y ejercicio",
        'diabetes_rec': "**Acciones urgentes:**\n- Consulte a un médico inmediatamente\n- Controle su dieta y medicación\n- Realice exámenes de glucosa",
        'risk_factors': "📊 Factores de Riesgo Clave",
        'categorical_risk': "Factores Categóricos de Riesgo",
        'disclaimer': "**Disclaimer:** Esta herramienta ofrece una estimación basada en datos, pero no reemplaza una evaluación médica profesional.",
        'sidebar_title': "📌 Información Adicional",
        'sidebar_content': """
- **Modelo**: XGBoost (Recall: 90%)
- **Fuente de datos**: CDC BRFSS 2015

**Factores Considerados:**
- Índice Salud General
- Actividad Física
- Hipertensión
- Dieta y colesterol
"""
    },
    'en': {
        'title': "🩺 Diabetes Risk Assessment Tool",
        'subtitle': "This tool predicts **diabetes risk** based on your health indicators.",
        'health_profile': "User Health Profile",
        'vital_signs': "Vital Signs",
        'lifestyle': "Lifestyle",
        'medical_history': "Medical History",
        'bmi': "Body Mass Index (BMI)",
        'gen_health': "General Health (1=Excellent, 5=Poor)",
        'demographic': "Demographic Information",
        'age': "Age Range",
        'education': "Education Level",
        'income': "Income Range",
        'gender': "Gender",
        'recent_health': "Recent Health Status",
        'mental_health': "Days of poor mental health (last 30 days)",
        'physical_health': "Days of poor physical health (last 30 days)",
        'predict_button': "🔍 Assess Diabetes Risk",
        'results_title': "🧾 Analysis Results",
        'no_diabetes': "## ✅ No diabetes",
        'prediabetes': "## ⚠️ Prediabetes",
        'diabetes': "## 🚨 Diabetes",
        'no_diabetes_rec': "**Tips:**\n- Maintain your healthy habits\n- Get checkups every 5 years",
        'prediabetes_rec': "**Recommendations:**\n- Consult a doctor for confirmation\n- Monitor your glucose levels\n- Adjust diet and exercise",
        'diabetes_rec': "**Urgent actions:**\n- Consult a doctor immediately\n- Control your diet and medication\n- Perform glucose tests",
        'risk_factors': "📊 Key Risk Factors",
        'categorical_risk': "Categorical Risk Factors",
        'disclaimer': "**Disclaimer:** This tool provides a data-based estimate but does not replace professional medical evaluation.",
        'sidebar_title': "📌 Additional Information",
        'sidebar_content': """
- **Model**: XGBoost (Recall: 90%)
- **Data source**: CDC BRFSS 2015

**Factors Considered:**
- General Health Index
- Physical Activity
- Hypertension
- Diet and cholesterol
"""
    }
}