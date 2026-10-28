
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "newsapp" / "saved_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

fake = pd.read_csv(BASE_DIR / "Fake.csv", nrows=3000)
true = pd.read_csv(BASE_DIR / "True.csv", nrows=3000)

fake["label"] = "fake"
true["label"] = "real"

data = pd.concat([fake, true], ignore_index=True)
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

X = data["title"] + " " + data["text"]
y = data["label"]

print("Preparing text data...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_vectorized = vectorizer.fit_transform(X)

print("Training model...")

model = LogisticRegression(max_iter=300)
model.fit(X_vectorized, y)

print("Saving model and vectorizer...")

joblib.dump(model, MODEL_DIR / "model.joblib")
joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")

print("Training complete!")
print(f"Saved files in: {MODEL_DIR}")
