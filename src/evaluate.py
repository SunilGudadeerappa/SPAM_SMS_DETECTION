import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


def evaluate_model():
    # Load TF-IDF features
    with open("data/features/X_tfidf.pkl", "rb") as f:
        X = pickle.load(f)

    with open("data/features/y_labels.pkl", "rb") as f:
        y = pickle.load(f)

    # Train-test split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load trained model
    with open("models/spam_classifier.pkl", "rb") as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()
