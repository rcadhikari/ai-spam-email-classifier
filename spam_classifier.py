"""
Spam Email Classifier
----------------------
A simple machine learning project that classifies text messages as spam or not spam (ham)
using Naive Bayes and CountVectorizer from scikit-learn.

Author: RC Adhikari
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Sample dataset
data = {
    'text': [
        "Win money now!",
        "Hey, are we still meeting tomorrow?",
        "Congratulations, you've won a prize!",
        "Can we reschedule our meeting?",
        "Claim your free vacation now!",
        "Let's catch up later"
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

# 2. Create DataFrame and map labels
df = pd.DataFrame(data)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# 4. Convert text to numerical vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test_vec)
print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 7. Predict on new email
new_email = ["Congratulations! You've been selected for a free iPhone"]
new_vec = vectorizer.transform(new_email)
prediction = model.predict(new_vec)

print("\n📧 New Email Prediction:", "Spam 🚨" if prediction[0] == 1 else "Ham ✅")
