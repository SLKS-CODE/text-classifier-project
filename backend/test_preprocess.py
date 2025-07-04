from preprocess import load_data

# Load cleaned and split data
X_train, X_test, y_train, y_test = load_data()

# Print first 5 samples
print("Sample Cleaned Tweets:")
for i in range(5):
    print(f"{i+1}. {X_train.iloc[i]}")
