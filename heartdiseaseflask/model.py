import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Collect and Clean the Data

# Load the dataset
url = 'heart.csv'  # Local path to your dataset file
df = pd.read_csv(url)

# Replace missing values marked as '?'
df.replace('?', pd.NA, inplace=True)

# Convert data types
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values using imputer (e.g., median strategy)
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Standardize the data
scaler = StandardScaler()
X = df_imputed.drop(columns='target')
y = df_imputed['target']
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Train the Models

# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),  # SVM needs probability=True for AUC-ROC
    'Decision Tree': DecisionTreeClassifier()
}

# Train the models
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# Step 3: Test and Evaluate the Models

# Evaluate the models on the test set
model_performance = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # Use average='macro' for multiclass
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # AUC-ROC score calculation
    if hasattr(model, 'predict_proba'):
        # For binary classification, get probabilities for the positive class (1)
        if len(model.classes_) == 2:  # Check if it's binary classification
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # Positive class probabilities
        else:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')  # One-vs-Rest for multiclass
    else:
        roc_auc = roc_auc_score(y_test, model.decision_function(X_test), multi_class='ovr')  # One-vs-Rest for multiclass

    model_performance[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': roc_auc
    }

# Convert the performance metrics into a DataFrame for easier comparison
performance_df = pd.DataFrame(model_performance).T
print(performance_df)

# Step 4: Identify Important Attributes (For Random Forest and Decision Tree)

# For Random Forest and Decision Tree, you can get feature importances
rf_model = trained_models['Random Forest']
importances = rf_model.feature_importances_

# Create a DataFrame of feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Step 5: Visualize the Performance of Different Models

performance_df.plot(kind='bar', figsize=(10, 7))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.show()
