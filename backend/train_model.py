import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from preprocess import load_data
import pickle
import tensorflow_hub as hub

# Step 1: Load Data
X_train, X_test, y_train, y_test = load_data()

# Step 2: Tokenization
MAX_VOCAB = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 50

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post')

# Step 3: Build the Model WITHOUT GloVe
model = Sequential([
    Embedding(MAX_VOCAB, EMBEDDING_DIM, input_length=MAX_LENGTH),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Negative, Neutral, Positive
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Step 4: Train the Model
model.fit(X_train_pad, y_train, epochs=15, batch_size=64, validation_data=(X_test_pad, y_test))

# Step 5: Save the Model
model.save("sentiment_model.h5")
print("✅ Model saved as sentiment_model.h5")
# Save tokenizer to file
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer saved as tokenizer.pkl")
# Load USE model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Encode text
X_train_embed = embed(X_train)
X_test_embed = embed(X_test)