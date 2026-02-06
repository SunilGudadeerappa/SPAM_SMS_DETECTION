import pandas as pd


import pandas as pd


def load_data():
    """
    STEP A: Structural cleaning
    - Load raw spam dataset
    - Keep only useful columns
    - Rename columns to standard names
    """

    # Correct relative path from project root
    file_path = "data/original/spam.csv"

    # Read dataset
    df = pd.read_csv(file_path, encoding="latin-1")

    # Keep only required columns
    df = df[['v1', 'v2']]

    # Rename columns
    df.columns = ['label', 'message']

    # Standardize labels
    df['label'] = df['label'].str.lower().str.strip()

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df['label'].value_counts())
