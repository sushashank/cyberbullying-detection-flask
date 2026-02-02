import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import re

# Step 1: Load the dataset
data_path = os.path.join(os.getcwd(), '..', 'data', 'cyberbullying_tweets(ML).csv')
df = pd.read_csv(data_path)

# Step 2: Preprocess the data
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    return text

df['tweet_text'] = df['tweet_text'].astype(str).apply(clean_text)

# Step 3: Create and fit the vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['tweet_text'])

# Step 4: Save the vectorizer
models_dir = os.path.join(os.getcwd(), '..', 'models')
os.makedirs(models_dir, exist_ok=True)

vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
with open(vectorizer_path, 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
print(f"✅ Vectorizer saved successfully at {vectorizer_path}!")