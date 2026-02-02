from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__, template_folder='templates')

# Global variables for the vectorizer and model
vectorizer = None
model = None

# Load the vectorizer and model at startup
try:
    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)

    # Load the vectorizer
    vectorizer_path = os.path.join(os.getcwd(), 'models', 'vectorizer.pkl')
    with open(vectorizer_path, 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    print("✅ Vectorizer loaded successfully!")

    # Load the model
    model_path = os.path.join(os.getcwd(), 'models', 'best_model.pkl')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model or vectorizer: {e}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text')
        if text:
            # Check if vectorizer and model are loaded
            if vectorizer is None or model is None:
                raise ValueError("Model or vectorizer not loaded")

            # List of specific words to trigger cyberbullying detection
            trigger_words = ["fuck", "hate", "rape", "dumb", "stupid", "kill you"]

            # Check if any of the trigger words are present in the input text (case-insensitive)
            if any(word in text.lower() for word in trigger_words):
                return render_template('index.html', prediction="Cyberbullying Detected!")

            # Transform the input text using the loaded vectorizer
            text_vector = vectorizer.transform([text])

            # Make prediction
            prediction = model.predict(text_vector)[0]

            # Convert prediction to readable output
            result = "Cyberbullying Detected!" if prediction == 1 else "No Cyberbullying Detected!"
            return render_template('index.html', prediction=result)

        else:
            return render_template('index.html', prediction="Please enter some text!")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return render_template('index.html', prediction="Error during prediction!")


if __name__ == '__main__':
    app.run(debug=True, port=5001)
