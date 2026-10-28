import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

fake = pd.read_csv(BASE_DIR / "Fake.csv")
true = pd.read_csv(BASE_DIR / "True.csv")

# Create labels
fake["label"] = "fake"
true["label"] = "real"

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Features and labels
X = data["title"] + " " + data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)


# Prediction function
def predict_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction