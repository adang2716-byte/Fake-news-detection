
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load only 3,000 rows from each dataset
fake = pd.read_csv(BASE_DIR / "Fake.csv", nrows=3000)
true = pd.read_csv(BASE_DIR / "True.csv", nrows=3000)

# Create labels
fake["label"] = "fake"
true["label"] = "real"

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Handle missing values
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Features and labels
X = data["title"] + " " + data["text"]
y = data["label"]

# Limit vocabulary to reduce memory usage
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    dtype="float32"
)

X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_vectorized, y)


def predict_news(text):
    text = str(text).strip()

    if not text:
        return "Please enter some news text."

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    return prediction
