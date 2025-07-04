# backend/train_model_USE.py

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from preprocess import load_data

# Step 1: Load Data
X_train_raw, X_test_raw, y_train, y_test = load_data()

# Step 2: Load USE (Universal Sentence Encoder) from TensorFlow Hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Step 3: Convert sentences to embeddings
X_train_embed = embed(X_train_raw).numpy()
X_test_embed = embed(X_test_raw).numpy()

# Step 4: Build Model (Dense only, no Embedding layer)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(512,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 5: Train the model
model.fit(X_train_embed, y_train, epochs=10, batch_size=32, validation_data=(X_test_embed, y_test))

# Step 6: Save model
model.save("sentiment_model_USE.h5")
print("✅ USE model saved as sentiment_model_USE.h5")
