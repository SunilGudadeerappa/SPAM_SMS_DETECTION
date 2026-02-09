import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def train_model():
    # Load features
    with open("data/features/X_tfidf.pkl", "rb") as f:
        X = pickle.load(f)

    with open("data/features/y_labels.pkl", "rb") as f:
        y = pickle.load(f)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate quickly
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model training completed")
    print("Accuracy:", accuracy)

    # Save model
    with open("models/spam_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved to models/spam_classifier.pkl")


if __name__ == "__main__":
    train_model()
