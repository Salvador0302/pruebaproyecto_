from utils import load_model

try:
    model = load_model()
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"Error loading model: {str(e)}")