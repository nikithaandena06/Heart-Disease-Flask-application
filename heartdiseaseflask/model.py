from flask import Flask, render_template, request, jsonify
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
app = Flask(__name__)

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

# Home route to serve the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Convert form data to the appropriate data types
    input_data = pd.DataFrame([data], columns=['age', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal'])
    input_data = input_data.astype(float)

    # Preprocess input data (handle missing values and scale)
    input_data.replace('?', np.nan, inplace=True)
    input_data = imputer.transform(input_data)
    input_data_scaled = scaler.transform(input_data)

    # Make predictions with all models
    predictions = {}
    for name, model in trained_models.items():
        predictions[name] = model.predict(input_data_scaled)[0]  # Predict for the first (and only) row

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
