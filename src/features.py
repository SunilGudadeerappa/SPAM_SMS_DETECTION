import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features():
    # Load cleaned data
    df = pd.read_csv("data/cleaned/clean_spam_sms.csv")

    X = df["message"]
    y = df["label"]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000
    )

    X_tfidf = vectorizer.fit_transform(X)

    # Save features and vectorizer
    with open("data/features/X_tfidf.pkl", "wb") as f:
        pickle.dump(X_tfidf, f)

    with open("data/features/y_labels.pkl", "wb") as f:
        pickle.dump(y, f)

    with open("data/features/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Feature extraction completed!")
    print("TF-IDF features shape:", X_tfidf.shape)


if __name__ == "__main__":
    extract_features()
