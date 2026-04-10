import joblib
import pandas as pd
from pathlib import Path

# Get the absolute path to the models directory (one level up from fastapi_app)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "decision_tree_v1.pkl"
ENCODINGS_PATH = BASE_DIR / "models" / "encodings.pkl"

model = None
encodings = None

def load_model():
    global model, encodings
    if model is None:
        model = joblib.load(MODEL_PATH)
        encodings = joblib.load(ENCODINGS_PATH)
        print("Model and encodings loaded successfully")
    return model, encodings