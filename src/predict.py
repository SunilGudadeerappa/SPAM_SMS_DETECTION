import pickle
import sys
import re


def clean_text(text):
    """
    Clean input SMS text (same logic as preprocessing)
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sms(message):
    # Load vectorizer
    with open("data/features/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Load trained model
    with open("models/spam_classifier.pkl", "rb") as f:
        model = pickle.load(f)

    # Clean and transform message
    message_cleaned = clean_text(message)
    message_vectorized = vectorizer.transform([message_cleaned])

    # Predict
    prediction = model.predict(message_vectorized)[0]

    return prediction


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"Your SMS message here\"")
        sys.exit(1)

    sms = sys.argv[1]
    result = predict_sms(sms)

    print("\nSMS:", sms)
    print("Prediction:", result.upper())
