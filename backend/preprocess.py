import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"#", "", text)        # remove hashtags
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    return text.strip()

def load_data():
    # Read 6-column CSV
    df = pd.read_csv('../dataset/training.csv', encoding='latin-1', header=None)

    # Only keep sentiment and tweet text columns
    df = df[[0, 5]]
    df.columns = ['sentiment', 'text']

    # Map sentiment: 0=Negative, 2=Neutral, 4=Positive
    df['sentiment'] = df['sentiment'].replace({0: 0, 2: 1, 4: 2})

    # Clean the text
    df['text'] = df['text'].apply(clean_text)

    # Use a sample (e.g., 50,000 rows)
    df = df.sample(50000, random_state=42)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment'], test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
