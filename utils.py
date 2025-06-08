import pickle
import pandas as pd

def load_model():
    """Load trained model from file"""
    with open('models/modelo_optimo_1.0_9_100_7_0.01_0.7_42.sav', 'rb') as f:
        return pickle.load(f)

def predict_diabetes(input_data):
    # Convertir input_data en DataFrame
    input_df = pd.DataFrame([input_data])

    # Renombrar columna si es necesario
    if 'Índice_de_Salud_General' in input_df.columns:
        input_df.rename(columns={'Índice_de_Salud_General': 'Health_Risk_Index'}, inplace=True)

    # Asegurar orden correcto de columnas
    expected_features = [
        'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
        'Income', 'Health_Risk_Index'
    ]

    input_df = input_df[expected_features]  # Reordenar de forma segura

    # Cargar modelo y predecir
    model = load_model()
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    return pred, proba

# Feature explanations for users
FEATURE_EXPLANATIONS = {
    'HighBP': "Do you have high blood pressure?",
    'HighChol': "Do you have high cholesterol?",
    'CholCheck': "Ha realizado un chequeo de colesterol en los últimos años?",
    'Smoker': "Es fumador?",
    'Stroke': "Tubo ataque cerebrovascular?",
    'HeartDiseaseorAttack': "Tubo enfermedad cardíaca o ataque al corazón?",
    'PhysActivity': "Realiza actividad fisica?",
    'Fruits': "Consume frutas?",
    'Veggies': "Consume verduras?",
    'HvyAlcoholConsump': "Consume alcohol?",
    'AnyHealthcare': "Tiene algún tipo de seguro o acceso a atención médica?",
    'NoDocbcCost': "Alguna vez no visitó al médico por razones de costo?",
    'GenHlth': "Salud General",
    'MentHlth': "Salud Mental",
    'PhysHlth': "Salud Fisica",
    'DiffWalk': "Tiene dificultad para caminar?",
    'Sex': "Cual es su genero?",
    'Age': "Edad",
    'Education': "Cual es su nivel de educacion?",
    'Income': "Ingreso",
    'Health_Risk_Index': "Please provide information for Health Risk Index."
}