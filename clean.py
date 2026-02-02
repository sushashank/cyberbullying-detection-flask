import pandas as pd
import re
import os

# Load the original dataset
file_path = 'data/cyberbullying_tweets(ML).csv'
df = pd.read_csv(file_path)

# Clean the tweet text by removing special characters, URLs, and offensive words
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply the cleaning function to the 'tweet_text' column
df['tweet_text'] = df['tweet_text'].astype(str).apply(clean_text)

# Ensure the output directory exists
os.makedirs('data', exist_ok=True)

# Save the cleaned dataset
cleaned_file_path = 'data/cleaned_cyberbullying_tweets.csv'
df.to_csv(cleaned_file_path, index=False)

print(f"✅ Cleaned data saved to {cleaned_file_path}")