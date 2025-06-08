import pickle
import pandas as pd

def load_model():
    """Load trained model from file"""
    with open('models/modelo_optimo_1.0_9_100_7_0.01_0.7_42.sav', 'rb') as f:
        return pickle.load(f)

def predict_diabetes(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Define expected features in EXACT order the model expects
    expected_features = [
        'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
        'Income', 'Índice_de_Salud_General'
    ]

    # Ensure correct feature order
    input_df = input_df[expected_features]
    
    # Load model and predict
    model = load_model()
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    return pred, proba

# Feature explanations for users
FEATURE_EXPLANATIONS = {
    'HighBP': "¿Tiene presión arterial alta?",
    'HighChol': "¿Tiene colesterol alto?",
    'CholCheck': "¿Se ha revisado el colesterol en los últimos 5 años?",
    'Smoker': "¿Ha fumado al menos 100 cigarrillos en su vida?",
    'Stroke': "¿Alguna vez le han diagnosticado un derrame cerebral?",
    'HeartDiseaseorAttack': "¿Tiene enfermedad coronaria o ha tenido un ataque al corazón?",
    'PhysActivity': "¿Realiza actividad física fuera de su trabajo habitual?",
    'Fruits': "¿Consume frutas 1 o más veces al día?",
    'Veggies': "¿Consume verduras 1 o más veces al día?",
    'HvyAlcoholConsump': "¿Es un bebedor excesivo de alcohol? (Hombres: >14 bebidas/semana, Mujeres: >7 bebidas/semana)",
    'AnyHealthcare': "¿Tiene algún tipo de seguro de salud?",
    'NoDocbcCost': "¿Alguna vez no visitó al médico por razones de costo?",
    'GenHlth': "¿Cómo calificaría su salud general? (1 = Excelente, 5 = Mala)",
    'MentHlth': "Días de mala salud mental en los últimos 30 días",
    'PhysHlth': "Días de mala salud física en los últimos 30 días",
    'DiffWalk': "¿Tiene dificultad para caminar o subir escaleras?",
    'Sex': "Sexo biológico",
    'Age': "Grupo de edad",
    'Education': "Nivel educativo más alto",
    'Income': "Rango de ingresos anuales",
    'Índice_de_Salud_General': "Índice combinado de salud general y BMI"
}