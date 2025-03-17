import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load your dataset (ensure your dataset is in the same directory or provide the full path)
data = pd.read_csv('customer_churn_data.csv')  # Replace with your dataset filename
print(data.head())

# Data Preprocessing
# Handle missing values
data = data.dropna()

# Encode categorical columns
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Feature selection
features = ['Age', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
X = data[features]
y = data['Exited']  # Target variable (Churn: 0 = No, 1 = Yes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model training with Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')

# Classification report
report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')


# Save the model
joblib.dump(model, 'customer_churn_model.pkl')
