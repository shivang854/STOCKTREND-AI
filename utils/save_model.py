import joblib
import os

def save_model(model, model_name):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")