
# 🧠 Sentiment Classifier Web App

A simple yet powerful sentiment analysis web app that predicts whether a given text expresses **Positive**, **Neutral**, or **Negative** sentiment. Built using **TensorFlow**, **Flask**, and the **Universal Sentence Encoder (USE)** from TensorFlow Hub.

---

## 📌 Features

- 🧾 Accepts user text input via a web interface  
- 🧠 Uses pre-trained Universal Sentence Encoder (USE) for sentence embedding  
- 📊 Predicts sentiment using a deep learning model (LSTM-based)  
- 💬 Returns confidence score with the predicted sentiment  
- ⚙️ Flask backend with a lightweight HTML frontend  
- 🔄 CORS-enabled API endpoint for frontend-backend integration

---

## 🛠️ Tech Stack

| Layer      | Technology                           |
|------------|--------------------------------------|
| Frontend   | HTML, CSS, JavaScript                |
| Backend    | Python, Flask, Flask-CORS            |
| Model      | TensorFlow, LSTM, Universal Sentence Encoder |
| Dataset    | Twitter Sentiment Dataset (Cleaned)  |

---

## 🚀 Project Structure

```
text-classifier-project/
│
├── backend/
│   ├── app.py                  # Flask server
│   ├── preprocess.py           # Data loading & preprocessing
│   ├── train_model_USE.py      # Model training using USE
│   ├── sentiment_model_USE.h5  # Trained model
│   └── tokenizer.pkl           # Tokenizer (if used with other models)
│
├── dataset/
│   └── training.csv            # Twitter sentiment dataset (locally used)
│
├── frontend/
│   └── index.html              # Simple UI for predictions
│
└── requirements.txt            # Python dependencies
```

---

## 💡 How It Works

1. User enters a sentence on the web page.
2. Text is sent to Flask backend via a `POST` request.
3. Backend uses the Universal Sentence Encoder to convert the text to a vector.
4. Pre-trained LSTM model predicts the sentiment class.
5. Response is sent back and displayed on the web page with confidence.

---

## 🖥️ Run Locally

> ⚠️ Make sure you have Python 3.9+ and pip installed.

```bash
# Step 1: Clone the repo
git clone https://github.com/SLKS-CODE/text-classifier-project.git
cd text-classifier-project/backend

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run Flask server
python app.py
```

Then, open `frontend/index.html` in your browser.

---

## 🌐 API Endpoint

```http
POST /predict
Content-Type: application/json
{
  "text": "I love this project!"
}
```

### ✅ Sample Response

```json
{
  "input": "I love this project!",
  "predicted_label": "Positive",
  "confidence": "98.01%"
}
```

---

## 🙌 Author

Created by [SLKS-CODE](https://github.com/SLKS-CODE)

---
