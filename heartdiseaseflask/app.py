from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Sample user and admin data (replace with a database in production)
users = {
    "user@example.com": "password123"
}
admins = {
    "admin@example.com": "adminpassword"
}

# Load and preprocess the dataset
url = 'heart.csv'
df = pd.read_csv(url)

# Data cleaning and preprocessing
df.replace('?', pd.NA, inplace=True)
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Standardize the data
scaler = StandardScaler()
X = df_imputed.drop(columns='target')
y = df_imputed['target']
X_scaled = scaler.fit_transform(X)

# Train the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier()
}

trained_models = {name: model.fit(X_scaled, y) for name, model in models.items()}

# Route to serve the index page (login page)
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the homepage after login
@app.route('/homepage', methods=['POST'])
def homepage():
    return render_template('homepage.html')
    # User login
    if email in users and users[email] == password:
        return render_template('homepage.html')

    # Admin login
    if email in admins and admins[email] == password:
        return render_template('homepage.html')

    # If login fails, redirect back to the index
    return redirect(url_for('homepage.html'))

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    print("Received request:", request.json)  # Debugging line
    data = request.json
    input_data = pd.DataFrame([data])

    input_data.replace('?', np.nan, inplace=True)
    input_data = imputer.transform(input_data)  # Fill missing values
    input_data_scaled = scaler.transform(input_data)  # Standardize

    predictions = {}
    for name, model in trained_models.items():
        predictions[name] = model.predict(input_data_scaled)[0]  # Predict for the first (and only) row

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
