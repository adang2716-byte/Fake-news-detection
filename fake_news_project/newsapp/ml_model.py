
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"

# Load the already-trained model and vectorizer
model = joblib.load(MODEL_DIR / "model.joblib")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")


def predict_news(text):
    text = str(text).strip()

    if not text:
        return "Please enter some news text."

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    return prediction
