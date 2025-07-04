# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Initialize Flask & enable CORS
app = Flask(__name__)
CORS(app)

# Load model and embedder
model = tf.keras.models.load_model("sentiment_model_USE.h5", custom_objects={'KerasLayer': hub.KerasLayer})
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Labels for output
labels = ["Negative", "Neutral", "Positive"]

# Home route
@app.route("/")
def home():
    return "🚀 USE Sentiment Classifier API is running!"

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    embedding = embed([text])
    prediction = model.predict(embedding)[0]
    label = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return jsonify({
        "input": text,
        "predicted_label": label,
        "confidence": f"{confidence:.2f}%"
    })

# Start the app
if __name__ == "__main__":
    app.run(debug=True)
