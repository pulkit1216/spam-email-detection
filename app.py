import joblib

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Take input from user
message = input("Enter your message: ")

# Convert message into numbers
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

# Show result
if prediction[0] == 1:
    print("🚨 This is SPAM message!")
else:
    print("✅ This is NOT spam (Ham)")