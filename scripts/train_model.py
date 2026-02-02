import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

# Step 3: Load the vectorizer
models_dir = os.path.join(os.getcwd(), '..', 'models')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

with open(vectorizer_path, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Step 4: Transform the data
X = vectorizer.transform(df['tweet_text'])
y = df['cyberbullying_type']  # Replace with the correct target column

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate the model (optional)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save the model
model_path = os.path.join(models_dir, 'best_model.pkl')
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)
print(f"✅ Model saved successfully at {model_path}!")