import pickle
import pandas as pd

def load_model():
    """Load trained model from file"""
    with open('models/modelo_optimo_1.0_9_100_7_0.01_0.7_42.sav', 'rb') as f:
        return pickle.load(f)

def predict_diabetes(input_data):
    """
    Make prediction using trained model
    input_data: Dictionary of user inputs
    Returns: (prediction, probability)
    """
    # Create feature DataFrame (MUST match training columns order)
    features = [
        'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
        'Income', 'Health_Risk_Index'
    ]
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_data], columns=features)
    
    # Load model and predict
    model = load_model()
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # Probability of diabetes
    
    return pred, proba

# Feature explanations for users
FEATURE_EXPLANATIONS = {
    'HighBP': "Do you have high blood pressure?",
    'HighChol': "Do you have high cholesterol?",
    # Add explanations for all other features...
}