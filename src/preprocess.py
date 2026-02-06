import re
from load_data import load_data


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data():
    df = load_data()

    # Clean text
    df["message"] = df["message"].apply(clean_text)

    # 🔴 IMPORTANT: remove empty / NaN messages
    df = df[df["message"].str.len() > 0]

    # Save cleaned data
    output_path = "data/cleaned/clean_spam_sms.csv"
    df.to_csv(output_path, index=False)

    print("Preprocessing completed!")
    print("Cleaned data shape:", df.shape)


if __name__ == "__main__":
    preprocess_data()
