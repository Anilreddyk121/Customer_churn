from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the saved model
model = joblib.load('customer_churn_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Create an index.html file for your form

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form (ensure the form has inputs matching your model features)
        age = float(request.form['age'])
        balance = float(request.form['balance'])
        num_products = int(request.form['num_products'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])

        # Prepare the features for prediction
        features = np.array([[age, balance, num_products, has_cr_card, is_active_member]])

        # Make prediction
        prediction = model.predict(features)

        # Return result
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
